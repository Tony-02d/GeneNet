use lib::models::{AppState, Dataset, Note, Project, Tab, RequestModel, ResponseModel, SubgraphNode, SubgraphEdge};
use eframe::egui;
use eframe::egui::{Align2, Color32, Pos2, Rect, Sense, Ui, Vec2};
use eframe::emath::RectTransform;
use egui::{Direction, Layout};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Bar, BarChart, Plot};
use polars::datatypes::BooleanChunked;
use polars::error::PolarsResult;
use polars::prelude::DataFrame;
use polars::prelude::{ChunkCompareEq, DataType};
use polars::prelude::{CsvReader, SerReader, CsvReadOptions};
use rfd::FileDialog;
use std::ops::Not;
use serde_json::json;
use crate::lib;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::fs::File;
use std::error::Error;
use poll_promise::Promise;
use std::collections::HashMap;
use std::thread;

fn update_debug_panel(state: &mut AppState, message: &str) {
    state.debug_output.push_str(message);
    state.debug_output.push('\n');
}

fn capture_stderr(child: &mut std::process::Child, state: &mut AppState) {
    if let Some(stderr) = child.stderr.take() {
        let mut reader = BufReader::new(stderr);
        let mut buffer = String::new();

        thread::spawn(move || {
            loop {
                buffer.clear();
                match reader.read_line(&mut buffer) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(mut state_lock) = state_mutex.lock() {
                            state_lock.debug_output.push_str(&buffer);
                        }
                    },
                    Err(_) => break,
                }
            }
        });
    }
}

lazy_static::lazy_static! {
    static ref state_mutex: std::sync::Mutex<AppState> = std::sync::Mutex::new(AppState {
        projects: Vec::new(),
        selected_project_path: Vec::new(),
        selected_tab: Tab::Preview,
        pending_delete: None,
        note_editing: None,
        column_to_remove: None,
        column_to_match: None,
        match_value: String::new(),
        graph_pan: Vec2::ZERO,
        graph_zoom: 1.0,
        node_positions: Vec::new(),
        layout_done: false,
        bar_genes: Vec::new(),
        bar_counts: Vec::new(),
        is_fetching: false,
        bar_selected_value: String::new(),
        bar_chart_promise: None,
        subgraph_gene_symbol: String::new(),
        is_fetching_subgraph: false,
        subgraph_promise: None,
        subgraph_nodes: Vec::new(),
        subgraph_edges: Vec::new(),
        subgraph_node_positions: HashMap::new(),
        subgraph_layout_done: false,
        subgraph_pan: Vec2::ZERO,
        subgraph_zoom: 1.0,
        top_genes_count: String::new(),
        is_fetching_top_genes: false,
        top_genes_promise: None,
        top_genes_nodes: Vec::new(),
        top_genes_edges: Vec::new(),
        top_genes_node_positions: HashMap::new(),
        top_genes_layout_done: false,
        top_genes_pan: Vec2::ZERO,
        top_genes_zoom: 1.0,
        predict_gene_symbol: String::new(),
        is_fetching_predict: false,
        is_training_model: false,
        predict_promise: None,
        predict_nodes: Vec::new(),
        predict_edges: Vec::new(),
        predict_edge_scores: HashMap::new(),
        predict_node_positions: HashMap::new(),
        predict_layout_done: false,
        predict_pan: Vec2::ZERO,
        predict_zoom: 1.0,
        predict_threshold: 0.9,
        predict_model_trained: false,
        debug_output: String::new(),
        debug_panel_height: 150.0,
        debug_panel_visible: true,
    });
}

fn request_bar_chart_blocking(
    df: &mut DataFrame,
    key: String,
    value: String,
    top_n: Option<usize>,
) -> Result<ResponseModel, Box<dyn Error>> {
    let python_executable = dotenv::var("PYTHON_EXECUTABLE").unwrap();
    let python_script = "backend/main.py";

    let mut child = match Command::new(python_executable)
        .arg(python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
            Ok(child) => {
                child
            },
            Err(e) => {
                let error_msg = format!("Failed to spawn Python process: {}", e);
                eprintln!("{}", error_msg);
                return Err(error_msg.into());
            }
        };

    let req = RequestModel::new("get_bar_chart_data")
        .with_parameters(json!({
            "key": key,
            "value": value,
            "top_n": top_n
        }))
        .with_dataframe(df)?;

    let req_json = serde_json::to_string(&req)? + "\n";

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(req_json.as_bytes())?;
        drop(stdin);
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("Python backend exited with error: {}", status).into());
    }
    let mut response = None;
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    match serde_json::from_str::<ResponseModel>(&line) {
                        Ok(resp) => {
                            response = Some(resp);
                            break;
                        },
                        Err(e) => {
                            eprintln!("Failed to parse response: {}", e);
                            return Err(format!("Failed to parse response: {}", e).into());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read line from Python process: {}", e);
                    return Err(format!("Failed to read line from Python process: {}", e).into());
                }
            }
        }
    }
    if let Some(resp) = response {
        Ok(resp)
    } else {
        Err("No response from Python backend".into())
    }
}

fn request_bar_chart_promise(
    df: DataFrame,
    key: String,
    value: String,
    top_n: Option<usize>,
) -> Promise<Result<ResponseModel, String>> {
    Promise::spawn_thread("bar_chart_request", move || {
        let mut df_clone = df;
        request_bar_chart_blocking(&mut df_clone, key, value, top_n)
            .map_err(|e| e.to_string())
    })
}

fn request_subgraph_blocking(
    df: &mut DataFrame,
    gene_symbol: String,
) -> Result<ResponseModel, Box<dyn Error>> {
    let python_executable = dotenv::var("PYTHON_EXECUTABLE").unwrap_or_else(|_| ".venv/bin/python".to_string());
    let python_script = "backend/main.py";

    let mut child = match Command::new(python_executable)
        .arg(python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn() {
            Ok(child) => {
                child
            },
            Err(e) => {
                eprintln!("Failed to spawn Python process: {}", e);
                return Err(format!("Failed to spawn Python process: {}", e).into());
            }
        };

    let req = RequestModel::new("get_subgraph_data")
        .with_parameters(json!({
            "gene_symbol": gene_symbol
        }))
        .with_dataframe(df)?;

    let req_json = serde_json::to_string(&req)? + "\n";

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(req_json.as_bytes())?;
        drop(stdin);
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("Python backend exited with error: {}", status).into());
    }
    let mut response = None;
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    match serde_json::from_str::<ResponseModel>(&line) {
                        Ok(resp) => {
                            response = Some(resp);
                            break;
                        },
                        Err(e) => {
                            eprintln!("Failed to parse response: {}", e);
                            return Err(format!("Failed to parse response: {}", e).into());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read line from Python process: {}", e);
                    return Err(format!("Failed to read line from Python process: {}", e).into());
                }
            }
        }
    }
    if let Some(resp) = response {
        Ok(resp)
    } else {
        Err("No response from Python backend".into())
    }
}

fn request_subgraph_promise(
    df: DataFrame,
    gene_symbol: String,
) -> Promise<Result<ResponseModel, String>> {
    Promise::spawn_thread("subgraph_request", move || {
        let mut df_clone = df;
        request_subgraph_blocking(&mut df_clone, gene_symbol)
            .map_err(|e| e.to_string())
    })
}

fn request_top_genes_graph_blocking(
    df: &mut DataFrame,
    top_n: usize,
) -> Result<ResponseModel, Box<dyn Error>> {
    let python_executable = dotenv::var("PYTHON_EXECUTABLE").unwrap_or_else(|_| ".venv/bin/python".to_string());
    let python_script = "backend/main.py";

    let mut child = match Command::new(python_executable)
        .arg(python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn() {
            Ok(child) => {
                child
            },
            Err(e) => {
                eprintln!("Failed to spawn Python process: {}", e);
                return Err(format!("Failed to spawn Python process: {}", e).into());
            }
        };

    let req = RequestModel::new("get_top_genes_graph")
        .with_parameters(json!({
            "top_n": top_n
        }))
        .with_dataframe(df)?;

    let req_json = serde_json::to_string(&req)? + "\n";

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(req_json.as_bytes())?;
        drop(stdin);
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("Python backend exited with error: {}", status).into());
    }
    let mut response = None;
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    match serde_json::from_str::<ResponseModel>(&line) {
                        Ok(resp) => {
                            response = Some(resp);
                            break;
                        },
                        Err(e) => {
                            eprintln!("Failed to parse response: {}", e);
                            return Err(format!("Failed to parse response: {}", e).into());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read line from Python process: {}", e);
                    return Err(format!("Failed to read line from Python process: {}", e).into());
                }
            }
        }
    }
    if let Some(resp) = response {
        Ok(resp)
    } else {
        Err("No response from Python backend".into())
    }
}

fn request_top_genes_graph_promise(
    df: DataFrame,
    top_n: usize,
) -> Promise<Result<ResponseModel, String>> {
    Promise::spawn_thread("top_genes_graph_request", move || {
        let mut df_clone = df;
        request_top_genes_graph_blocking(&mut df_clone, top_n)
            .map_err(|e| e.to_string())
    })
}

fn request_predict_blocking(
    df: &mut DataFrame,
    gene_symbol: String,
    is_training: bool,
    threshold: f32,
) -> Result<ResponseModel, Box<dyn Error>> {
    let python_executable = dotenv::var("PYTHON_EXECUTABLE").unwrap_or_else(|_| ".venv/bin/python".to_string());
    let python_script = "backend/main.py";

    let mut child = match Command::new(python_executable)
        .arg(python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn() {
            Ok(child) => {
                child
            },
            Err(e) => {
                eprintln!("Failed to spawn Python process: {}", e);
                return Err(format!("Failed to spawn Python process: {}", e).into());
            }
        };

    let action = if is_training { "train_prediction_model" } else { "predict_gene_links" };

    let req = RequestModel::new(action)
        .with_parameters(json!({
            "gene_symbol": gene_symbol,
            "threshold": threshold
        }))
        .with_dataframe(df)?;

    let req_json = serde_json::to_string(&req)? + "\n";

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(req_json.as_bytes())?;
        drop(stdin);
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("Python backend exited with error: {}", status).into());
    }
    let mut response = None;
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    match serde_json::from_str::<ResponseModel>(&line) {
                        Ok(resp) => {
                            response = Some(resp);
                            break;
                        },
                        Err(e) => {
                            eprintln!("Failed to parse response: {}", e);
                            return Err(format!("Failed to parse response: {}", e).into());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read line from Python process: {}", e);
                    return Err(format!("Failed to read line from Python process: {}", e).into());
                }
            }
        }
    }
    if let Some(resp) = response {
        Ok(resp)
    } else {
        Err("No response from Python backend".into())
    }
}

fn request_predict_promise(
    df: DataFrame,
    gene_symbol: String,
    is_training: bool,
    threshold: f32,
) -> Promise<Result<ResponseModel, String>> {
    Promise::spawn_thread("predict_request", move || {
        let mut df_clone = df;
        request_predict_blocking(&mut df_clone, gene_symbol, is_training, threshold)
            .map_err(|e| e.to_string())
    })
}

pub fn get_project_mut<'a>(projects: &'a mut [Project], path: &[usize]) -> Option<&'a mut Project> {
    if path.is_empty() {
        return None;
    }
    let first = path[0];
    if first >= projects.len() {
        return None;
    }
    let proj = &mut projects[first];
    if path.len() == 1 {
        Some(proj)
    } else {
        get_project_mut(&mut proj.subprojects, &path[1..])
    }
}

fn delete_project_at_path(projects: &mut Vec<Project>, path: &[usize]) {
    if path.is_empty() {
        return;
    }
    if path.len() == 1 {
        let idx = path[0];
        if idx < projects.len() {
            projects.remove(idx);
        }
    } else {
        let parent_path = &path[..path.len() - 1];
        if let Some(parent) = get_project_mut(projects, parent_path) {
            let idx = path[path.len() - 1];
            if idx < parent.subprojects.len() {
                parent.subprojects.remove(idx);
            }
        }
    }
}

fn draw_project_tree(
    ui: &mut egui::Ui,
    proj: &Project,
    path_so_far: &Vec<usize>,
    level: usize,
    state: &AppState,
) -> Option<(Vec<usize>, Option<usize>)> {
    let mut clicked: Option<(Vec<usize>, Option<usize>)> = None;

    let is_selected = state.selected_project_path == *path_so_far;

    ui.horizontal(|ui| {
        ui.add_space(level as f32 * 16.0);
        if ui.selectable_label(is_selected, &proj.name).clicked() {
            clicked = Some((path_so_far.clone(), None));
        }
    });

    for (note_idx, note) in proj.notes.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.add_space((level + 1) as f32 * 16.0);
            if ui.button(format!("📝 {}", note.name)).clicked() {
                clicked = Some((path_so_far.clone(), Some(note_idx)));
            }
        });
    }

    if !proj.datasets.is_empty() {
        for ds in &proj.datasets {
            ui.horizontal(|ui| {
                ui.add_space((level + 1) as f32 * 16.0);
                ui.label(&ds.name);
            });
        }
    }

    for (child_idx, child_proj) in proj.subprojects.iter().enumerate() {
        let mut child_path = path_so_far.clone();
        child_path.push(child_idx);
        if clicked.is_none() {
            if let Some(found) = draw_project_tree(ui, child_proj, &child_path, level + 1, state) {
                clicked = Some(found);
            }
        }
    }

    clicked
}

pub fn side_panel(ui: &mut Ui, state: &mut AppState) {
    let mut maybe_clicked: Option<(Vec<usize>, Option<usize>)> = None;

    for (i, proj) in state.projects.iter().enumerate() {
        let initial_path = vec![i];
        if maybe_clicked.is_none() {
            maybe_clicked = draw_project_tree(ui, proj, &initial_path, 0, state);
        }
    }

    if let Some((new_path, maybe_note_idx)) = maybe_clicked {
        state.selected_project_path = new_path.clone();

        if let Some(note_idx) = maybe_note_idx {
            state.note_editing = Some((new_path, note_idx));
        } else {
            state.note_editing = None;
        }
    }

    if !state.selected_project_path.is_empty() {
        ui.horizontal(|ui| {
            ui.label("Project name:");
            if let Some(sel_proj) =
                get_project_mut(&mut state.projects, &state.selected_project_path)
            {
                ui.text_edit_singleline(&mut sel_proj.name);
            }
        });

        ui.add_space(8.0);

        if ui.button("＋ Add Note").clicked() {
            if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
                proj.notes.push(Note {
                    name: "New Note".into(),
                    content: "".into(),
                });
                let note_idx = proj.notes.len() - 1;
                state.note_editing = Some((state.selected_project_path.clone(), note_idx));
            }
        }
    }

    if ui.button("＋ Add Root Project").clicked() {
        state.projects.push(Project {
            name: "New".into(),
            subprojects: Vec::new(),
            datasets: Vec::new(),
            selected_dataset: None,
            notes: Vec::new(),
        });
        let idx = state.projects.len() - 1;
        state.selected_project_path = vec![idx];
    }

    if !state.selected_project_path.is_empty() {
        ui.add_space(4.0);
        if ui.button("＋ Add Subproject").clicked() {
            if let Some(parent) = get_project_mut(&mut state.projects, &state.selected_project_path)
            {
                parent.subprojects.push(Project {
                    name: "New".into(),
                    subprojects: Vec::new(),
                    datasets: Vec::new(),
                    selected_dataset: None,
                    notes: Vec::new(),
                });
                let child_idx = parent.subprojects.len() - 1;
                let mut new_path = state.selected_project_path.clone();
                new_path.push(child_idx);
                state.selected_project_path = new_path;
            }
        }
    }

    if !state.selected_project_path.is_empty() {
        ui.add_space(8.0);
        if ui.button("🗑 Delete Project").clicked() {
            state.pending_delete = Some(state.selected_project_path.clone());
        }
    }

    ui.separator();

    if let Some(path_to_delete) = state.pending_delete.clone() {
        egui::Window::new("Confirm Deletion")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        "Are you sure you want to delete this project\nand all of its subprojects?",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Yes, Delete").clicked() {
                            delete_project_at_path(&mut state.projects, &path_to_delete);
                            state.selected_project_path.clear();
                            state.pending_delete = None;
                        }
                        if ui.button("Cancel").clicked() {
                            state.pending_delete = None;
                        }
                    });
                });
            });
    }
}

pub fn tab_bar(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        for &tab in &[Tab::Preview, Tab::BarChart, Tab::Subgraph, Tab::TopGenesGraph, Tab::Predict] {
            let label = format!("{:?}", tab);
            if ui
                .selectable_label(state.selected_tab == tab, label)
                .clicked()
            {
                state.selected_tab = tab;
            }
        }
    });
}

pub fn preview_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            ui.vertical(|ui| {
                ui.heading("Datasets");
                for (ds_idx, ds) in proj.datasets.iter().enumerate() {
                    if ui
                        .selectable_label(proj.selected_dataset == Some(ds_idx), &ds.name)
                        .clicked()
                    {
                        proj.selected_dataset = Some(ds_idx);
                        state.column_to_remove = None;
                        state.column_to_match = None;
                        state.match_value.clear();
                    }
                }

                if ui.button("＋ Add CSV").clicked() {
                    if let Some(path) = FileDialog::new().add_filter("CSV", &["csv"]).pick_file() {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("csv")
                            .to_string();

                        let file_result = File::open(&path);
                        if let Err(err) = file_result {
                            ui.colored_label(egui::Color32::RED, format!("File open error: {}", err));
                            return;
                        }

                        let result: PolarsResult<DataFrame> =
                            CsvReader::new(file_result.unwrap())
                                .with_options(CsvReadOptions::default()
                                    .with_has_header(true)
                                    .with_infer_schema_length(Some(10_000))
                                    .with_ignore_errors(true)
                                )
                                .finish();

                        if let Ok(df) = result {
                            proj.datasets.push(Dataset {
                                name: name.clone(),
                                path: path.clone(),
                                df: Some(df),
                            });
                            proj.selected_dataset = Some(proj.datasets.len() - 1);
                            state.column_to_remove = None;
                            state.column_to_match = None;
                            state.match_value.clear();
                            ctx.request_repaint();
                        }
                    }
                }

                ui.label(format!("({} loaded)", proj.datasets.len()));
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get_mut(ds_idx) {
                    if let Some(df) = &ds.df {
                        let col_names: Vec<String> = df
                            .get_column_names()
                            .iter()
                            .map(|s| s.to_string())
                            .collect();

                        ui.horizontal(|ui| {
                            ui.label("Filter (remove one column):");

                            let selected_col = state.column_to_remove.get_or_insert_with(|| {
                                col_names.get(0).cloned().unwrap_or_default()
                            });

                            egui::ComboBox::from_id_source("remove_column_combo")
                                .selected_text(selected_col.clone())
                                .show_ui(ui, |ui| {
                                    for name in &col_names {
                                        ui.selectable_value(
                                            selected_col,
                                            name.clone(),
                                            name.clone(),
                                        );
                                    }
                                });

                            if ui.button("Remove Column").clicked() {
                                if !selected_col.is_empty() && col_names.contains(selected_col) {
                                    if let Some(orig_df) = &mut ds.df {
                                        if let Ok(mut new_df) = orig_df.drop(selected_col) {
                                            *orig_df = new_df.clone();
                                            state.column_to_remove = None;
                                            ctx.request_repaint();
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Remove rows where");

                            let selected_match_col =
                                state.column_to_match.get_or_insert_with(|| {
                                    col_names.get(0).cloned().unwrap_or_default()
                                });

                            egui::ComboBox::from_id_source("remove_rows_combo")
                                .selected_text(selected_match_col.clone())
                                .show_ui(ui, |ui| {
                                    for name in &col_names {
                                        ui.selectable_value(
                                            selected_match_col,
                                            name.clone(),
                                            name.clone(),
                                        );
                                    }
                                });

                            ui.label("= ");

                            ui.add(
                                egui::TextEdit::singleline(&mut state.match_value)
                                    .id_source("match_value_input")
                                    .hint_text("value to match"),
                            );

                            if ui.button("Remove Rows").clicked() {
                                let input_str = state.match_value.trim().to_string();
                                if !input_str.is_empty() && col_names.contains(selected_match_col) {
                                    if let Some(orig_df) = &mut ds.df {
                                        if let Ok(s) = orig_df
                                            .column(selected_match_col)
                                            .and_then(|series| series.cast(&DataType::String))
                                        {
                                            if let Ok(str_chunked) = s.str() {
                                                let equal_mask: BooleanChunked =
                                                    str_chunked.equal(input_str.as_str());
                                                let keep_mask = equal_mask.not();

                                                if let Ok(filtered_df) = orig_df.filter(&keep_mask)
                                                {
                                                    *orig_df = filtered_df;
                                                    state.column_to_match = None;
                                                    state.match_value.clear();
                                                    ctx.request_repaint();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();
                    }
                }
            }

            ui.vertical(|ui| {
                egui::ScrollArea::both().show(ui, |ui| {
                    if let Some(ds_idx) = proj.selected_dataset {
                        if let Some(ds) = proj.datasets.get(ds_idx) {
                            if let Some(df) = &ds.df {
                                let cols = df.get_columns();
                                let rows = df.height();

                                let mut builder = TableBuilder::new(ui).striped(true).cell_layout(
                                    Layout::centered_and_justified(Direction::LeftToRight),
                                );

                                for _ in cols.iter() {
                                    builder = builder.column(Column::auto());
                                }

                                let table = builder.header(20.0, |mut header| {
                                    for series in cols.iter() {
                                        header.col(|ui| {
                                            ui.heading(series.name().to_string());
                                        });
                                    }
                                });

                                table.body(|body| {
                                    body.rows(18.0, rows, |mut row| {
                                        let row_idx = row.index();
                                        for series in cols.iter() {
                                            let val = series.get(row_idx);
                                            row.col(|ui| {
                                                ui.label(format!("{:?}", val.unwrap().get_str().unwrap_or("—")));
                                            });
                                        }
                                    });
                                });
                            } else {
                                ui.label("Dataset loaded but empty.");
                            }
                        }
                    } else if proj.datasets.is_empty() {
                        ui.label("No dataset loaded.");
                    } else {
                        ui.label("Select a dataset to preview.");
                    }
                });
            });
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Select or create a project first.");
        });
    }
}

pub fn subgraph_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            if proj.datasets.is_empty() {
                ui.label("No datasets loaded. Go to Preview → ＋ Add CSV to load one.");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Gene Symbol:");
                let mut gene_symbol = state.subgraph_gene_symbol.clone();
                if ui.text_edit_singleline(&mut gene_symbol).changed() {
                    state.subgraph_gene_symbol = gene_symbol;
                }

                let df_for_request_opt = if let Some(ds_idx) = proj.selected_dataset {
                    if let Some(ds) = proj.datasets.get(ds_idx) {
                        ds.df.clone()
                    } else {
                        None
                    }
                } else {
                    None
                };

                if ui.button("Generate Subgraph").clicked() {
                    if state.is_fetching_subgraph {
                    } else if state.subgraph_promise.is_some() {
                    } else {
                        if state.subgraph_gene_symbol.trim().is_empty() {
                            ui.colored_label(egui::Color32::RED, "Please enter a Gene Symbol.");
                        } else {
                            if let Some(df_ref) = df_for_request_opt {
                                state.is_fetching_subgraph = true;

                                let df_for_request = df_ref.clone();
                                let gene_symbol = state.subgraph_gene_symbol.clone();

                                state.subgraph_promise = Some(request_subgraph_promise(
                                    df_for_request,
                                    gene_symbol,
                                ));
                                ctx.request_repaint();
                            } else {
                                ui.colored_label(egui::Color32::RED, "No dataset found. Please select a dataset before generating a subgraph.");
                            }
                        }
                    }
                }
            });

            ui.separator();

            let current_idx = proj.selected_dataset.unwrap_or(0);
            let mut chosen_idx = current_idx;
            ui.horizontal(|ui| {
                ui.label("Dataset: ");
                egui::ComboBox::from_id_source("subgraph_dataset_combo")
                    .selected_text(
                        proj.datasets
                            .get(chosen_idx)
                            .map(|ds| ds.name.clone())
                            .unwrap_or_else(|| "<none>".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, ds) in proj.datasets.iter().enumerate() {
                            ui.selectable_value(&mut chosen_idx, i, ds.name.clone());
                        }
                    });
                if chosen_idx != current_idx {
                    proj.selected_dataset = Some(chosen_idx);
                }
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    if let Some(df_ref) = &ds.df {
                        if state.is_fetching_subgraph {
                            ui.label("Generating subgraph...");
                        } else if state.subgraph_promise.is_some() {
                            ui.label("Processing request...");
                        } else {
                            if state.subgraph_nodes.is_empty() {
                                ui.label("Enter a Gene Symbol and click 'Generate Subgraph' to create a subgraph visualization.");
                            } else {
                                let available_size = ui.available_size();
                                let (rect, response) = ui.allocate_exact_size(available_size, Sense::drag().union(Sense::hover()));

                                if response.dragged() {
                                    state.subgraph_pan += response.drag_delta();
                                }

                                if response.hovered() {
                                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                                    if scroll != 0.0 {
                                        let zoom_factor = (scroll * 0.005).exp();
                                        state.subgraph_zoom = (state.subgraph_zoom * zoom_factor).clamp(0.1, 10.0);
                                    }
                                }

                                if !state.subgraph_layout_done && !state.subgraph_nodes.is_empty() {
                                    let node_count = state.subgraph_nodes.len();
                                    let mut positions = HashMap::new();

                                    let gene_node = state.subgraph_nodes.iter().find(|node| node.node_type == "gene" && node.label == state.subgraph_gene_symbol);

                                    if let Some(gene_node) = gene_node {
                                        positions.insert(gene_node.id.clone(), Vec2::ZERO);
                                    } else {
                                        if let Some(any_gene) = state.subgraph_nodes.iter().find(|node| node.node_type == "gene") {
                                            positions.insert(any_gene.id.clone(), Vec2::ZERO);
                                        }
                                    }

                                    let radius = 600.0;
                                    let remaining_nodes: Vec<_> = state.subgraph_nodes.iter()
                                        .filter(|node| gene_node.as_ref().map_or(true, |gn| node.id != gn.id))
                                        .collect();

                                    let angle_step = if remaining_nodes.is_empty() {
                                        0.0
                                    } else {
                                        2.0 * std::f32::consts::PI / remaining_nodes.len() as f32
                                    };

                                    for (i, node) in remaining_nodes.iter().enumerate() {
                                        let angle = i as f32 * angle_step;
                                        let x = radius * angle.cos();
                                        let y = radius * angle.sin();
                                        positions.insert(node.id.clone(), Vec2::new(x, y));
                                    }

                                    let area = 1_000.0 * 1_000.0;
                                    let k = (area / node_count as f32).sqrt();
                                    let iterations = 100;
                                    let mut disp = HashMap::new();

                                    for node in &state.subgraph_nodes {
                                        disp.insert(node.id.clone(), Vec2::ZERO);
                                    }

                                    for _ in 0..iterations {
                                        for node in &state.subgraph_nodes {
                                            disp.insert(node.id.clone(), Vec2::ZERO);
                                        }

                                        for i in 0..node_count {
                                            for j in (i+1)..node_count {
                                                let node1 = &state.subgraph_nodes[i];
                                                let node2 = &state.subgraph_nodes[j];

                                                let pos1 = positions.get(&node1.id).unwrap_or(&Vec2::ZERO);
                                                let pos2 = positions.get(&node2.id).unwrap_or(&Vec2::ZERO);

                                                let delta = *pos1 - *pos2;
                                                let dist = delta.length().max(1.0);
                                                let force = (k * k) / dist * 1.0;
                                                let direction = delta / dist;

                                                *disp.entry(node1.id.clone()).or_insert(Vec2::ZERO) += direction * force;
                                                *disp.entry(node2.id.clone()).or_insert(Vec2::ZERO) -= direction * force;
                                            }
                                        }

                                        for edge in &state.subgraph_edges {
                                            if let (Some(pos1), Some(pos2)) = (positions.get(&edge.source), positions.get(&edge.target)) {
                                                let delta = *pos1 - *pos2;
                                                let dist = delta.length().max(1.0);
                                                let force = (dist * dist) / k * 1.2;
                                                let direction = delta / dist;

                                                *disp.entry(edge.source.clone()).or_insert(Vec2::ZERO) -= direction * force;
                                                *disp.entry(edge.target.clone()).or_insert(Vec2::ZERO) += direction * force;
                                            }
                                        }

                                        let temperature = 100.0;
                                        for node in &state.subgraph_nodes {
                                            if let Some(d) = disp.get(&node.id) {
                                                let length = d.length().max(1.0);
                                                let displacement = (*d / length) * length.min(temperature);

                                                if let Some(pos) = positions.get_mut(&node.id) {
                                                    *pos += displacement;
                                                }
                                            }
                                        }
                                    }

                                    state.subgraph_node_positions = positions;
                                    state.subgraph_layout_done = true;
                                }

                                if state.subgraph_layout_done {
                                    let top_left = (-state.subgraph_pan) / state.subgraph_zoom;
                                    let world_size = rect.size() / state.subgraph_zoom;
                                    let world_rect = Rect::from_min_size(Pos2::new(top_left.x, top_left.y), world_size);
                                    let to_screen: RectTransform = RectTransform::from_to(world_rect, rect);
                                    let painter = ui.painter_at(rect);

                                    for edge in &state.subgraph_edges {
                                        if let (Some(pos1), Some(pos2)) = (
                                            state.subgraph_node_positions.get(&edge.source),
                                            state.subgraph_node_positions.get(&edge.target)
                                        ) {
                                            let sp1 = to_screen.transform_pos(Pos2::new(pos1.x, pos1.y));
                                            let sp2 = to_screen.transform_pos(Pos2::new(pos2.x, pos2.y));
                                            painter.line_segment([sp1, sp2], (0.6, Color32::WHITE));
                                        }
                                    }

                                    let base_node_radius_world = 25.0;
                                    for node in &state.subgraph_nodes {
                                        if let Some(pos) = state.subgraph_node_positions.get(&node.id) {
                                            let screen_pos = to_screen.transform_pos(Pos2::new(pos.x, pos.y));

                                            let node_radius_world = if node.node_type == "gene" {
                                                base_node_radius_world * 2.0
                                            } else {
                                                base_node_radius_world
                                            };

                                            let r_screen = node_radius_world * state.subgraph_zoom;

                                            let node_color = match node.node_type.as_str() {
                                                "gene" => {
                                                    Color32::from_rgb(50, 100, 255)
                                                },
                                                "disease" => Color32::from_rgb(200, 100, 100),
                                                _ => Color32::from_rgb(150, 150, 150),
                                            };

                                            painter.circle_filled(screen_pos, r_screen, node_color);

                                            let mut text_style = egui::TextStyle::Body.resolve(ui.style());
                                            text_style.size = 12.0;

                                            let offsets = [
                                                Vec2::new(-2.0, -2.0),
                                                Vec2::new(-2.0, 2.0),
                                                Vec2::new(2.0, -2.0),
                                                Vec2::new(2.0, 2.0),
                                                Vec2::new(0.0, -2.0),
                                                Vec2::new(0.0, 2.0),
                                                Vec2::new(-2.0, 0.0),
                                                Vec2::new(2.0, 0.0),
                                            ];

                                            for &off in &offsets {
                                                let pos = screen_pos + off;
                                                painter.text(
                                                    pos,
                                                    Align2::CENTER_CENTER,
                                                    &node.label,
                                                    text_style.clone(),
                                                    Color32::BLACK,
                                                );
                                            }

                                            painter.text(
                                                screen_pos,
                                                Align2::CENTER_CENTER,
                                                &node.label,
                                                text_style.clone(),
                                                Color32::WHITE,
                                            );

                                        }
                                    }
                                }
                            }
                        }

                        let promise_result = if let Some(promise) = &state.subgraph_promise {
                            let ready = promise.ready().cloned();
                            ready
                        } else {
                            None
                        };

                        if let Some(result) = promise_result {
                            state.subgraph_promise = None;
                            state.is_fetching_subgraph = false;

                            match result {
                                Ok(response) => {
                                    if let Some(result) = &response.result {

                                        if let Some(nodes) = result.get("nodes").and_then(|v| v.as_array()) {
                                            state.subgraph_nodes.clear();
                                            for node in nodes {
                                                if let (Some(id), Some(label), Some(node_type)) = (
                                                    node.get("id").and_then(|v| v.as_str()),
                                                    node.get("label").and_then(|v| v.as_str()),
                                                    node.get("type").and_then(|v| v.as_str())
                                                ) {
                                                    state.subgraph_nodes.push(lib::models::SubgraphNode {
                                                        id: id.to_string(),
                                                        label: label.to_string(),
                                                        node_type: node_type.to_string(),
                                                    });
                                                }
                                            }
                                        }

                                        if let Some(edges) = result.get("edges").and_then(|v| v.as_array()) {
                                            state.subgraph_edges.clear();
                                            for edge in edges {
                                                if let (Some(source), Some(target)) = (
                                                    edge.get("source").and_then(|v| v.as_str()),
                                                    edge.get("target").and_then(|v| v.as_str())
                                                ) {
                                                    state.subgraph_edges.push(lib::models::SubgraphEdge {
                                                        source: source.to_string(),
                                                        target: target.to_string(),
                                                    });

                                                }
                                            }
                                        }

                                        state.subgraph_layout_done = false;

                                        ctx.request_repaint();
                                    } else if let Some(error) = &response.error {
                                        ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                                    }
                                }
                                Err(e) => {
                                    ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                                }
                            }

                            ctx.request_repaint();
                        }
                    }
                }
            }
        });
    } else {
        ui.label("No project selected. Please select or create a project first.");
    }
}

pub fn top_genes_graph_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            if proj.datasets.is_empty() {
                ui.label("No datasets loaded. Go to Preview → ＋ Add CSV to load one.");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Top");
                let mut top_n_text = state.top_genes_count.clone();
                if ui.text_edit_singleline(&mut top_n_text).changed() {
                    state.top_genes_count = top_n_text;
                }
                ui.label("genes:");

                let df_for_request_opt = if let Some(ds_idx) = proj.selected_dataset {
                    if let Some(ds) = proj.datasets.get(ds_idx) {
                        ds.df.clone()
                    } else {
                        None
                    }
                } else {
                    None
                };

                if ui.button("Generate Graph").clicked() {
                    if state.is_fetching_top_genes {
                    } else if state.top_genes_promise.is_some() {
                    } else {
                        if state.top_genes_count.trim().is_empty() {
                            state.top_genes_count = "10".to_string();
                        }

                        if let Ok(n) = state.top_genes_count.parse::<usize>() {
                            if let Some(df_ref) = df_for_request_opt {
                                state.is_fetching_top_genes = true;

                                let df_for_request = df_ref.clone();

                                state.top_genes_promise = Some(request_top_genes_graph_promise(
                                    df_for_request,
                                    n,
                                ));
                                ctx.request_repaint();
                            } else {
                                ui.colored_label(egui::Color32::RED, "No dataset found. Please select a dataset before generating a graph.");
                            }
                        } else {
                            ui.colored_label(egui::Color32::RED, "Invalid input. Please enter a valid number.");
                        }
                    }
                }
            });

            ui.separator();

            let current_idx = proj.selected_dataset.unwrap_or(0);
            let mut chosen_idx = current_idx;
            ui.horizontal(|ui| {
                ui.label("Dataset: ");
                egui::ComboBox::from_id_source("top_genes_dataset_combo")
                    .selected_text(
                        proj.datasets
                            .get(chosen_idx)
                            .map(|ds| ds.name.clone())
                            .unwrap_or_else(|| "<none>".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, ds) in proj.datasets.iter().enumerate() {
                            ui.selectable_value(&mut chosen_idx, i, ds.name.clone());
                        }
                    });
                if chosen_idx != current_idx {
                    proj.selected_dataset = Some(chosen_idx);
                    state.top_genes_nodes.clear();
                    state.top_genes_edges.clear();
                    state.top_genes_node_positions.clear();
                    state.top_genes_layout_done = false;
                }
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    if let Some(df_ref) = &ds.df {
                        if state.is_fetching_top_genes {
                            ui.label("Generating graph...");
                        } else if state.top_genes_promise.is_some() {
                            ui.label("Processing request...");
                        } else {
                            if state.top_genes_nodes.is_empty() {
                                ui.label("Enter a number and click 'Generate Graph' to create a graph visualization of the top genes by disease count.");
                            } else {
                                let available_size = ui.available_size();
                                let (rect, response) = ui.allocate_exact_size(available_size, Sense::drag().union(Sense::hover()));

                                if response.dragged() {
                                    state.top_genes_pan += response.drag_delta();
                                }

                                if response.hovered() {
                                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                                    if scroll != 0.0 {
                                        let zoom_factor = (scroll * 0.005).exp();
                                        state.top_genes_zoom = (state.top_genes_zoom * zoom_factor).clamp(0.1, 10.0);
                                    }
                                }

                                if !state.top_genes_layout_done && !state.top_genes_nodes.is_empty() {
                                    let node_count = state.top_genes_nodes.len();
                                    let mut positions = HashMap::new();

                                    let gene_radius = 1.0;
                                    let disease_radius = 30.0;
                                    let center = Vec2::new(0.0, 0.0);

                                    let gene_nodes: Vec<&SubgraphNode> = state.top_genes_nodes.iter()
                                        .filter(|n| n.node_type == "gene")
                                        .collect();
                                    let disease_nodes: Vec<&SubgraphNode> = state.top_genes_nodes.iter()
                                        .filter(|n| n.node_type == "disease")
                                        .collect();

                                    let gene_count = gene_nodes.len();
                                    for (i, node) in gene_nodes.iter().enumerate() {
                                        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (gene_count.max(1) as f32);
                                        let pos = center + Vec2::new(angle.cos(), angle.sin()) * gene_radius;
                                        positions.insert(node.id.clone(), pos);
                                    }

                                    let disease_count = disease_nodes.len();
                                    for (i, node) in disease_nodes.iter().enumerate() {
                                        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (disease_count.max(1) as f32);
                                        let pos = center + Vec2::new(angle.cos(), angle.sin()) * disease_radius;
                                        positions.insert(node.id.clone(), pos);
                                    }

                                    let iterations = 200;
                                    for _ in 0..iterations {
                                        let mut disp = HashMap::new();

                                        for i in 0..node_count {
                                            for j in 0..node_count {
                                                if i == j {
                                                    continue;
                                                }

                                                let node_i = &state.top_genes_nodes[i];
                                                let node_j = &state.top_genes_nodes[j];

                                                if let (Some(pos_i), Some(pos_j)) = (positions.get(&node_i.id), positions.get(&node_j.id)) {
                                                    let delta = *pos_i - *pos_j;
                                                    let dist = delta.length().max(0.01);

                                                    let force = if node_i.node_type == "disease" && node_j.node_type == "disease" {
                                                        500_000.0 / (dist * dist)
                                                    } else if node_i.node_type == "gene" && node_j.node_type == "gene" {
                                                        1_000.0 / (dist * dist)
                                                    } else {
                                                        100_000.0 / (dist * dist)
                                                    };

                                                    let direction = delta / dist;

                                                    *disp.entry(node_i.id.clone()).or_insert(Vec2::ZERO) += direction * force;
                                                }
                                            }
                                        }

                                        let k = (1.0 / (node_count as f32).sqrt()) * 100.0;

                                        for edge in &state.top_genes_edges {
                                            if let (Some(pos1), Some(pos2)) = (positions.get(&edge.source), positions.get(&edge.target)) {
                                                let delta = *pos1 - *pos2;
                                                let dist = delta.length().max(1.0);

                                                let source_node = state.top_genes_nodes.iter().find(|n| n.id == edge.source);
                                                let target_node = state.top_genes_nodes.iter().find(|n| n.id == edge.target);

                                                let force = if let (Some(source), Some(target)) = (source_node, target_node) {
                                                    if source.node_type == "gene" && target.node_type == "disease" {
                                                        (dist * dist) / k * 0.01
                                                    } else if source.node_type == "disease" && target.node_type == "disease" {
                                                        (dist * dist) / k * 2.5
                                                    } else { // gene -> gene
                                                        (dist * dist) / k * 0.01
                                                    }
                                                } else {
                                                    (dist * dist) / k * 1.0
                                                };

                                                let direction = delta / dist;

                                                *disp.entry(edge.source.clone()).or_insert(Vec2::ZERO) -= direction * force;
                                                *disp.entry(edge.target.clone()).or_insert(Vec2::ZERO) += direction * force;
                                            }
                                        }

                                        for node in &state.top_genes_nodes {
                                            if let Some(d) = disp.get(&node.id) {
                                                let length = d.length().max(1.0);

                                                let node_temperature = if node.node_type == "gene" {
                                                    80.0
                                                } else {
                                                    150.0
                                                };

                                                let displacement = (*d / length) * length.min(node_temperature);

                                                if let Some(pos) = positions.get_mut(&node.id) {
                                                    *pos += displacement;
                                                }
                                            }
                                        }
                                    }

                                    state.top_genes_node_positions = positions;
                                    state.top_genes_layout_done = true;
                                }

                                if state.top_genes_layout_done {
                                    let top_left = (-state.top_genes_pan) / state.top_genes_zoom;
                                    let world_size = rect.size() / state.top_genes_zoom;
                                    let world_rect = Rect::from_min_size(Pos2::new(top_left.x, top_left.y), world_size);
                                    let to_screen: RectTransform = RectTransform::from_to(world_rect, rect);
                                    let painter = ui.painter_at(rect);

                                    for edge in &state.top_genes_edges {
                                        if let (Some(pos1), Some(pos2)) = (
                                            state.top_genes_node_positions.get(&edge.source),
                                            state.top_genes_node_positions.get(&edge.target)
                                        ) {
                                            let sp1 = to_screen.transform_pos(Pos2::new(pos1.x, pos1.y));
                                            let sp2 = to_screen.transform_pos(Pos2::new(pos2.x, pos2.y));
                                            painter.line_segment([sp1, sp2], (0.4, Color32::WHITE));
                                        }
                                    }

                                    let base_node_radius_world = 10.0;
                                    for node in &state.top_genes_nodes {
                                        if let Some(pos) = state.top_genes_node_positions.get(&node.id) {
                                            let screen_pos = to_screen.transform_pos(Pos2::new(pos.x, pos.y));

                                            let node_radius_world = if node.node_type == "gene" {
                                                base_node_radius_world * 3.0
                                            } else {
                                                base_node_radius_world
                                            };

                                            let r_screen = node_radius_world * state.top_genes_zoom;

                                            let node_color = match node.node_type.as_str() {
                                                "gene" => {
                                                    Color32::from_rgb(50, 100, 255)
                                                },
                                                "disease" => Color32::from_rgb(200, 100, 100),
                                                _ => Color32::GRAY,
                                            };

                                            painter.circle_filled(screen_pos, r_screen, node_color);

                                            let font_size = 10.0;
                                            let text_color = Color32::WHITE;
                                            painter.text(
                                                screen_pos,
                                                Align2::CENTER_CENTER,
                                                &node.label,
                                                egui::TextStyle::Body.resolve(ui.style()),
                                                text_color,
                                            );
                                        }
                                    }

                                    let legend_rect = Rect::from_min_size(
                                        rect.min + egui::vec2(10.0, 10.0),
                                        egui::vec2(150.0, 60.0),
                                    );
                                    painter.rect_filled(legend_rect, 5.0, Color32::from_black_alpha(180));

                                    let gene_color = Color32::from_rgb(50, 100, 255);
                                    let disease_color = Color32::from_rgb(200, 100, 100);

                                    let text_pos_y = legend_rect.min.y + 15.0;
                                    let circle_pos_x = legend_rect.min.x + 20.0;
                                    let text_pos_x = circle_pos_x + 25.0;

                                    painter.circle_filled(Pos2::new(circle_pos_x, text_pos_y), 8.0, gene_color);
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y),
                                        Align2::LEFT_CENTER,
                                        "Gene",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );

                                    painter.circle_filled(Pos2::new(circle_pos_x, text_pos_y + 25.0), 8.0, disease_color);
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y + 25.0),
                                        Align2::LEFT_CENTER,
                                        "Disease",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );
                                }
                            }
                        }

                        let promise_result = if let Some(promise) = &state.top_genes_promise {
                            let ready = promise.ready().cloned();
                            ready
                        } else {
                            None
                        };

                        if let Some(result) = promise_result {
                            state.top_genes_promise = None;
                            state.is_fetching_top_genes = false;

                            match result {
                                Ok(response) => {
                                    if let Some(result) = &response.result {
                                        if let Some(nodes) = result.get("nodes").and_then(|v| v.as_array()) {
                                            state.top_genes_nodes.clear();
                                            for node in nodes {
                                                if let (Some(id), Some(label), Some(node_type)) = (
                                                    node.get("id").and_then(|v| v.as_str()),
                                                    node.get("label").and_then(|v| v.as_str()),
                                                    node.get("type").and_then(|v| v.as_str()),
                                                ) {
                                                    state.top_genes_nodes.push(SubgraphNode {
                                                        id: id.to_string(),
                                                        label: label.to_string(),
                                                        node_type: node_type.to_string(),
                                                    });
                                                }
                                            }
                                        }

                                        if let Some(edges) = result.get("edges").and_then(|v| v.as_array()) {
                                            state.top_genes_edges.clear();
                                            for edge in edges {
                                                if let (Some(source), Some(target)) = (
                                                    edge.get("source").and_then(|v| v.as_str()),
                                                    edge.get("target").and_then(|v| v.as_str()),
                                                ) {
                                                    state.top_genes_edges.push(SubgraphEdge {
                                                        source: source.to_string(),
                                                        target: target.to_string(),
                                                    });
                                                }
                                            }
                                        }

                                        state.top_genes_layout_done = false;
                                    } else if let Some(error) = &response.error {
                                        ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                                    }
                                },
                                Err(e) => {
                                    ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        ui.label("No project selected. Please select or create a project first.");
    }
}

pub fn predict_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            if proj.datasets.is_empty() {
                ui.label("No datasets loaded. Go to Preview → ＋ Add CSV to load one.");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Gene Symbol:");
                let mut gene_symbol = state.predict_gene_symbol.clone();
                if ui.text_edit_singleline(&mut gene_symbol).changed() {
                    state.predict_gene_symbol = gene_symbol;
                }

                let df_for_request_opt = if let Some(ds_idx) = proj.selected_dataset {
                    if let Some(ds) = proj.datasets.get(ds_idx) {
                        ds.df.clone()
                    } else {
                        None
                    }
                } else {
                    None
                };

                if ui.button("Train Model").clicked() {
                    if state.is_fetching_predict || state.is_training_model {
                    } else if state.predict_promise.is_some() {
                    } else {
                        if state.predict_gene_symbol.trim().is_empty() {
                            ui.colored_label(egui::Color32::RED, "Please enter a Gene Symbol.");
                        } else {
                            if let Some(ref df_ref) = df_for_request_opt {
                                state.is_training_model = true;

                                let df_for_request = df_ref.clone();
                                let gene_symbol = state.predict_gene_symbol.clone();

                                state.predict_promise = Some(request_predict_promise(
                                    df_for_request,
                                    gene_symbol,
                                    true,
                                    state.predict_threshold,
                                ));
                                ctx.request_repaint();
                            } else {
                                ui.colored_label(egui::Color32::RED, "No dataset found. Please select a dataset before training.");
                            }
                        }
                    }
                }

                if ui.button("Predict").clicked() {
                    if state.is_fetching_predict || state.is_training_model {
                    } else if state.predict_promise.is_some() {
                    } else {
                        if !state.predict_model_trained {
                            ui.colored_label(egui::Color32::RED, "Please train the model first.");
                        } else if state.predict_gene_symbol.trim().is_empty() {
                            ui.colored_label(egui::Color32::RED, "Please enter a Gene Symbol.");
                        } else {
                            if let Some(ref df_ref) = df_for_request_opt {
                                state.is_fetching_predict = true;

                                let df_for_request = df_ref.clone();
                                let gene_symbol = state.predict_gene_symbol.clone();

                                state.predict_promise = Some(request_predict_promise(
                                    df_for_request,
                                    gene_symbol,
                                    false,
                                    state.predict_threshold,
                                ));
                                ctx.request_repaint();
                            } else {
                                ui.colored_label(egui::Color32::RED, "No dataset found. Please select a dataset before predicting.");
                            }
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Prediction Threshold:");
                let mut threshold_str = state.predict_threshold.to_string();
                if ui.text_edit_singleline(&mut threshold_str).changed() {
                    if let Ok(threshold) = threshold_str.parse::<f32>() {
                        if threshold > 0.0 && threshold <= 1.0 {
                            state.predict_threshold = threshold;
                        }
                    }
                }
                ui.label("(0.0 - 1.0)");
            });

            ui.separator();

            let current_idx = proj.selected_dataset.unwrap_or(0);
            let mut chosen_idx = current_idx;
            ui.horizontal(|ui| {
                ui.label("Dataset: ");
                egui::ComboBox::from_id_source("predict_dataset_combo")
                    .selected_text(
                        proj.datasets
                            .get(chosen_idx)
                            .map(|ds| ds.name.clone())
                            .unwrap_or_else(|| "<none>".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, ds) in proj.datasets.iter().enumerate() {
                            ui.selectable_value(&mut chosen_idx, i, ds.name.clone());
                        }
                    });
                if chosen_idx != current_idx {
                    proj.selected_dataset = Some(chosen_idx);
                    state.predict_model_trained = false;
                    state.predict_nodes.clear();
                    state.predict_edges.clear();
                    state.predict_edge_scores.clear();
                    state.predict_node_positions.clear();
                    state.predict_layout_done = false;
                }
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    if let Some(df_ref) = &ds.df {
                        if state.is_training_model {
                            ui.label("Training model...");
                        } else if state.is_fetching_predict {
                            ui.label("Predicting links...");
                        } else if state.predict_promise.is_some() {
                            ui.label("Processing request...");
                        } else {
                            if state.predict_nodes.is_empty() {
                                ui.label("Enter a Gene Symbol, train the model, and then click 'Predict' to see predicted gene-disease links.");
                            } else {
                                let available_size = ui.available_size();
                                let (rect, response) = ui.allocate_exact_size(available_size, Sense::drag().union(Sense::hover()));

                                if response.dragged() {
                                    state.predict_pan += response.drag_delta();
                                }

                                if response.hovered() {
                                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                                    if scroll != 0.0 {
                                        let zoom_factor = (scroll * 0.005).exp();
                                        state.predict_zoom = (state.predict_zoom * zoom_factor).clamp(0.1, 10.0);
                                    }
                                }

                                if !state.predict_layout_done && !state.predict_nodes.is_empty() {
                                    let node_count = state.predict_nodes.len();
                                    let mut positions = HashMap::new();

                                    let gene_radius = 30.0;
                                    let disease_radius = 40.0;
                                    let center = Vec2::new(0.0, 0.0);

                                    let gene_nodes: Vec<&SubgraphNode> = state.predict_nodes.iter()
                                        .filter(|n| n.node_type == "gene")
                                        .collect();
                                    let disease_nodes: Vec<&SubgraphNode> = state.predict_nodes.iter()
                                        .filter(|n| n.node_type == "disease")
                                        .collect();

                                    let gene_count = gene_nodes.len();
                                    for (i, node) in gene_nodes.iter().enumerate() {
                                        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (gene_count.max(1) as f32);
                                        let pos = center + Vec2::new(angle.cos(), angle.sin()) * gene_radius;
                                        positions.insert(node.id.clone(), pos);
                                    }

                                    let disease_count = disease_nodes.len();
                                    for (i, node) in disease_nodes.iter().enumerate() {
                                        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (disease_count.max(1) as f32);
                                        let pos = center + Vec2::new(angle.cos(), angle.sin()) * disease_radius;
                                        positions.insert(node.id.clone(), pos);
                                    }

                                    let iterations = 200;
                                    for _ in 0..iterations {
                                        let mut disp = HashMap::new();

                                        for i in 0..node_count {
                                            for j in 0..node_count {
                                                if i == j {
                                                    continue;
                                                }

                                                let node_i = &state.predict_nodes[i];
                                                let node_j = &state.predict_nodes[j];

                                                if let (Some(pos_i), Some(pos_j)) = (positions.get(&node_i.id), positions.get(&node_j.id)) {
                                                    let delta = *pos_i - *pos_j;
                                                    let dist = delta.length().max(0.01);

                                                    let force = if node_i.node_type == "disease" && node_j.node_type == "disease" {
                                                        600_000.0 / (dist * dist)
                                                    } else if node_i.node_type == "gene" && node_j.node_type == "gene" {
                                                        5_000.0 / (dist * dist)
                                                    } else {
                                                        10_000.0 / (dist * dist)
                                                    };

                                                    let direction = delta / dist;

                                                    *disp.entry(node_i.id.clone()).or_insert(Vec2::ZERO) += direction * force;
                                                }
                                            }
                                        }

                                        let k = (1.0 / (node_count as f32).sqrt()) * 100.0;

                                        for edge in &state.predict_edges {
                                            if let (Some(pos1), Some(pos2)) = (positions.get(&edge.source), positions.get(&edge.target)) {
                                                let delta = *pos1 - *pos2;
                                                let dist = delta.length().max(1.0);

                                                let source_node = state.predict_nodes.iter().find(|n| n.id == edge.source);
                                                let target_node = state.predict_nodes.iter().find(|n| n.id == edge.target);

                                                let force = if let (Some(source), Some(target)) = (source_node, target_node) {
                                                    if source.node_type == "gene" && target.node_type == "disease" {
                                                        (dist * dist) / k * 1.5
                                                    } else if source.node_type == "disease" && target.node_type == "gene" {
                                                        (dist * dist) / k * 1.5
                                                    } else {
                                                        (dist * dist) / k * 1.0
                                                    }
                                                } else {
                                                    (dist * dist) / k * 1.0
                                                };

                                                let direction = delta / dist;

                                                *disp.entry(edge.source.clone()).or_insert(Vec2::ZERO) -= direction * force;
                                                *disp.entry(edge.target.clone()).or_insert(Vec2::ZERO) += direction * force;
                                            }
                                        }

                                        for node in &state.predict_nodes {
                                            if let Some(d) = disp.get(&node.id) {
                                                let length = d.length().max(1.0);

                                                let node_temperature = if node.node_type == "gene" {
                                                    150.0
                                                } else {
                                                    80.0
                                                };

                                                let displacement = (*d / length) * length.min(node_temperature);

                                                if let Some(pos) = positions.get_mut(&node.id) {
                                                    *pos += displacement;
                                                }
                                            }
                                        }
                                    }

                                    state.predict_node_positions = positions;
                                    state.predict_layout_done = true;
                                }

                                if state.predict_layout_done {
                                    let top_left = (-state.predict_pan) / state.predict_zoom;
                                    let world_size = rect.size() / state.predict_zoom;
                                    let world_rect = Rect::from_min_size(Pos2::new(top_left.x, top_left.y), world_size);
                                    let to_screen: RectTransform = RectTransform::from_to(world_rect, rect);
                                    let painter = ui.painter_at(rect);

                                    for edge in &state.predict_edges {
                                        if let (Some(pos1), Some(pos2)) = (
                                            state.predict_node_positions.get(&edge.source),
                                            state.predict_node_positions.get(&edge.target)
                                        ) {
                                            let sp1 = to_screen.transform_pos(Pos2::new(pos1.x, pos1.y));
                                            let sp2 = to_screen.transform_pos(Pos2::new(pos2.x, pos2.y));

                                            let edge_key = format!("{}-{}", edge.source, edge.target);
                                            let edge_color = if let Some(score) = state.predict_edge_scores.get(&edge_key) {
                                                let intensity = (score * 255.0) as u8;
                                                Color32::from_rgb(255 - intensity, intensity, 0)
                                            } else {
                                                Color32::WHITE
                                            };

                                            painter.line_segment([sp1, sp2], (1.5, edge_color));

                                            if let Some(score) = state.predict_edge_scores.get(&edge_key) {
                                                let mid_point = Pos2::new((sp1.x + sp2.x) / 2.0, (sp1.y + sp2.y) / 2.0);
                                                let score_text = format!("{:.2}", score);
                                                painter.text(
                                                    mid_point,
                                                    Align2::CENTER_CENTER,
                                                    &score_text,
                                                    egui::TextStyle::Body.resolve(ui.style()),
                                                    Color32::WHITE,
                                                );
                                            }
                                        }
                                    }

                                    let base_node_radius_world = 8.0;
                                    for node in &state.predict_nodes {
                                        if let Some(pos) = state.predict_node_positions.get(&node.id) {
                                            let screen_pos = to_screen.transform_pos(Pos2::new(pos.x, pos.y));

                                            let node_radius_world = if node.node_type == "gene" {
                                                base_node_radius_world * 2.0
                                            } else {
                                                base_node_radius_world
                                            };

                                            let r_screen = node_radius_world * state.predict_zoom;

                                            let node_color = match node.node_type.as_str() {
                                                "gene" => {
                                                    Color32::from_rgb(50, 100, 255)
                                                },
                                                "disease" => Color32::from_rgb(200, 100, 100),
                                                _ => Color32::GRAY,
                                            };

                                            painter.circle_filled(screen_pos, r_screen, node_color);

                                            let font_size = 8.0;
                                            let text_color = Color32::WHITE;
                                            painter.text(
                                                screen_pos,
                                                Align2::CENTER_CENTER,
                                                &node.label,
                                                egui::TextStyle::Body.resolve(ui.style()),
                                                text_color,
                                            );
                                        }
                                    }

                                    let legend_rect = Rect::from_min_size(
                                        rect.min + egui::vec2(10.0, 10.0),
                                        egui::vec2(150.0, 100.0),
                                    );
                                    painter.rect_filled(legend_rect, 5.0, Color32::from_black_alpha(180));

                                    let gene_color = Color32::from_rgb(50, 100, 255);
                                    let disease_color = Color32::from_rgb(200, 100, 100);
                                    let high_score_color = Color32::from_rgb(0, 255, 0);
                                    let low_score_color = Color32::from_rgb(255, 0, 0);

                                    let text_pos_y = legend_rect.min.y + 15.0;
                                    let circle_pos_x = legend_rect.min.x + 20.0;
                                    let text_pos_x = circle_pos_x + 25.0;

                                    painter.circle_filled(Pos2::new(circle_pos_x, text_pos_y), 8.0, gene_color);
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y),
                                        Align2::LEFT_CENTER,
                                        "Gene",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );

                                    painter.circle_filled(Pos2::new(circle_pos_x, text_pos_y + 25.0), 8.0, disease_color);
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y + 25.0),
                                        Align2::LEFT_CENTER,
                                        "Disease",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );

                                    painter.line_segment(
                                        [Pos2::new(circle_pos_x - 10.0, text_pos_y + 50.0), Pos2::new(circle_pos_x + 10.0, text_pos_y + 50.0)],
                                        (3.0, high_score_color),
                                    );
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y + 50.0),
                                        Align2::LEFT_CENTER,
                                        "High Score",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );

                                    painter.line_segment(
                                        [Pos2::new(circle_pos_x - 10.0, text_pos_y + 75.0), Pos2::new(circle_pos_x + 10.0, text_pos_y + 75.0)],
                                        (3.0, low_score_color),
                                    );
                                    painter.text(
                                        Pos2::new(text_pos_x, text_pos_y + 75.0),
                                        Align2::LEFT_CENTER,
                                        "Low Score",
                                        egui::TextStyle::Body.resolve(ui.style()),
                                        Color32::WHITE,
                                    );
                                }
                            }
                        }

                        let promise_result = if let Some(promise) = &state.predict_promise {
                            let ready = promise.ready().cloned();
                            ready
                        } else {
                            None
                        };

                        if let Some(result) = promise_result {
                            state.predict_promise = None;

                            let was_training = state.is_training_model;
                            state.is_training_model = false;
                            state.is_fetching_predict = false;

                            match result {
                                Ok(response) => {
                                    if let Some(result) = &response.result {
                                        if was_training {
                                            if let Some(success) = result.get("success").and_then(|v| v.as_bool()) {
                                                if success {
                                                    state.predict_model_trained = true;
                                                    ui.colored_label(egui::Color32::GREEN, "Model trained successfully!");
                                                } else {
                                                    ui.colored_label(egui::Color32::RED, "Model training failed.");
                                                }
                                            }
                                        } else {
                                            if let Some(nodes) = result.get("nodes").and_then(|v| v.as_array()) {
                                                state.predict_nodes.clear();
                                                for node in nodes {
                                                    if let (Some(id), Some(label), Some(node_type)) = (
                                                        node.get("id").and_then(|v| v.as_str()),
                                                        node.get("label").and_then(|v| v.as_str()),
                                                        node.get("type").and_then(|v| v.as_str()),
                                                    ) {
                                                        state.predict_nodes.push(SubgraphNode {
                                                            id: id.to_string(),
                                                            label: label.to_string(),
                                                            node_type: node_type.to_string(),
                                                        });
                                                    }
                                                }
                                            }

                                            if let Some(edges) = result.get("edges").and_then(|v| v.as_array()) {
                                                state.predict_edges.clear();
                                                state.predict_edge_scores.clear();
                                                for edge in edges {
                                                    if let (Some(source), Some(target), Some(score)) = (
                                                        edge.get("source").and_then(|v| v.as_str()),
                                                        edge.get("target").and_then(|v| v.as_str()),
                                                        edge.get("score").and_then(|v| v.as_f64()),
                                                    ) {
                                                        let source_str = source.to_string();
                                                        let target_str = target.to_string();

                                                        state.predict_edges.push(SubgraphEdge {
                                                            source: source_str.clone(),
                                                            target: target_str.clone(),
                                                        });

                                                        let edge_key = format!("{}-{}", source_str, target_str);
                                                        state.predict_edge_scores.insert(edge_key, score as f32);
                                                    }
                                                }
                                            }

                                            state.predict_layout_done = false;
                                        }
                                    } else if let Some(error) = &response.error {
                                        ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                                    }
                                },
                                Err(e) => {
                                    ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        ui.label("No project selected. Please select or create a project first.");
    }
}

pub fn bar_chart_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            if proj.datasets.is_empty() {
                ui.label("No datasets loaded. Go to Preview → ＋ Add CSV to load one.");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Top");
                let mut top_n_text = state.bar_selected_value.clone();
                if ui.text_edit_singleline(&mut top_n_text).changed() {
                    state.bar_selected_value = top_n_text;
                }
                ui.label("genes:");

                let df_for_request_opt = if let Some(ds_idx) = proj.selected_dataset {
                    if let Some(ds) = proj.datasets.get(ds_idx) {
                        ds.df.clone()
                    } else {
                        None
                    }
                } else {
                    None
                };

                if ui.button("Update").clicked() {
                    if state.is_fetching {
                    } else if state.bar_chart_promise.is_some() {
                    } else {
                        if state.bar_selected_value.trim().is_empty() {
                            state.bar_selected_value = "10".to_string();
                        }

                        if let Ok(n) = state.bar_selected_value.parse::<usize>() {
                            if let Some(df_ref) = df_for_request_opt {
                                state.is_fetching = true;

                                let df_for_request = df_ref.clone();

                                state.bar_chart_promise = Some(request_bar_chart_promise(
                                    df_for_request,
                                    "GeneSymbol".to_string(),
                                    n.to_string(),
                                    Some(n),
                                ));
                                ctx.request_repaint();
                            } else {
                                ui.colored_label(egui::Color32::RED, "No dataset found. Please select a dataset before updating.");
                            }
                        } else {
                            ui.colored_label(egui::Color32::RED, "Invalid input. Please enter a valid number.");
                        }
                    }
                }
            });

            ui.separator();

            let current_idx = proj.selected_dataset.unwrap_or(0);
            let mut chosen_idx = current_idx;
            ui.horizontal(|ui| {
                ui.label("Dataset: ");
                egui::ComboBox::from_id_source("bar_dataset_combo")
                    .selected_text(
                        proj.datasets
                            .get(chosen_idx)
                            .map(|ds| ds.name.clone())
                            .unwrap_or_else(|| "<none>".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, ds) in proj.datasets.iter().enumerate() {
                            ui.selectable_value(&mut chosen_idx, i, ds.name.clone());
                        }
                    });
                if chosen_idx != current_idx {
                    proj.selected_dataset = Some(chosen_idx);
                    state.bar_genes.clear();
                    state.bar_counts.clear();
                }
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    if let Some(df_ref) = &ds.df {
                        if state.is_fetching {
                            ui.label("Loading chart data...");
                        } else if state.bar_chart_promise.is_some() {
                            ui.label("Processing request...");
                        } else if state.bar_genes.is_empty() {
                            ui.label("Click 'Update' to load data.");
                        }

                        let promise_result = if let Some(promise) = &state.bar_chart_promise {
                            let ready = promise.ready().cloned();
                            ready
                        } else {
                            None
                        };

                        if let Some(result) = promise_result {
                            state.bar_chart_promise = None;
                            state.is_fetching = false;

                            match result {
                                Ok(response) => {
                                    if let Some(result) = &response.result {
                                        if let Some(sorted_bar_data) = result.get("sorted_bar_data").and_then(|v| v.as_array()) {
                                            state.bar_genes.clear();
                                            state.bar_counts.clear();

                                            for item in sorted_bar_data {
                                                if let Some(gene) = item.get("gene").and_then(|v| v.as_str()) {
                                                    if let Some(count) = item.get("count").and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))) {
                                                        state.bar_genes.push(gene.to_string());
                                                        state.bar_counts.push(count);
                                                    }
                                                }
                                            }
                                        }
                                        else {
                                            ui.colored_label(egui::Color32::RED, "Unexpected JSON: missing 'sorted_bar_data'");
                                        }
                                    } else {
                                        ui.colored_label(egui::Color32::RED, "Response missing 'result'");
                                    }
                                }
                                Err(err) => {
                                    ui.colored_label(egui::Color32::RED, format!("Request error: {}", err));
                                }
                            }

                            ctx.request_repaint();
                        } else {
                            ctx.request_repaint_after(std::time::Duration::from_millis(500));
                        }

                        if !state.bar_genes.is_empty() && state.bar_genes.len() == state.bar_counts.len() {
                            let bars: Vec<Bar> = state.bar_counts
                                .iter()
                                .enumerate()
                                .map(|(i, &c)| Bar::new(i as f64, c))
                                .collect();

                            Plot::new("bar_chart")
                                .x_axis_formatter(|x, _| {
                                    let idx = x.value.round() as usize;
                                    state.bar_genes.get(idx).cloned().unwrap_or_default()
                                })
                                .show(ui, |plot_ui| {
                                    plot_ui.bar_chart(BarChart::new(bars));
                                });
                        }
                    } else {
                        ui.label("Dataset is empty.");
                    }
                }
            } else {
                ui.label("Select a dataset.");
            }
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Select or create a project first.");
        });
    }
}
