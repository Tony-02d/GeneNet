mod lib {
    include!("mod.rs");
}

use crate::lib::models::{AppState, Tab};
use crate::lib::ui;
use crate::lib::ui::get_project_mut;
use dotenv::dotenv;
use eframe::egui;
use eframe::egui::{Vec2, Visuals};
use std::error::Error;
use std::io::Read;

pub struct MyApp {
    state: AppState,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            state: AppState {
                projects: vec![],
                selected_project_path: vec![],
                selected_tab: Tab::Preview,
                pending_delete: None,
                note_editing: None,
                column_to_remove: None,
                column_to_match: None,
                match_value: "".to_string(),
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
                subgraph_node_positions: std::collections::HashMap::new(),
                subgraph_layout_done: false,
                subgraph_pan: Vec2::ZERO,
                subgraph_zoom: 1.0,
                top_genes_count: "10".to_string(),
                is_fetching_top_genes: false,
                top_genes_promise: None,
                top_genes_nodes: Vec::new(),
                top_genes_edges: Vec::new(),
                top_genes_node_positions: std::collections::HashMap::new(),
                top_genes_layout_done: false,
                top_genes_pan: Vec2::ZERO,
                top_genes_zoom: 1.0,
                predict_gene_symbol: String::new(),
                is_fetching_predict: false,
                is_training_model: false,
                predict_promise: None,
                predict_nodes: Vec::new(),
                predict_edges: Vec::new(),
                predict_edge_scores: std::collections::HashMap::new(),
                predict_node_positions: std::collections::HashMap::new(),
                predict_layout_done: false,
                predict_pan: Vec2::ZERO,
                predict_zoom: 1.0,
                predict_threshold: 0.9,
                predict_model_trained: false,
                debug_output: String::new(),
                debug_panel_height: 150.0,
                debug_panel_visible: true,
            },
        }
    }
}

fn check_debug_output(state: &mut AppState) {
    if let Ok(file) = std::fs::File::open("debug_output.log") {
        use std::io::{BufRead, BufReader, Seek, SeekFrom};

        let mut reader = BufReader::new(file);

        let mut contents = String::new();
        let _ = reader.read_to_string(&mut contents);

        if !contents.is_empty() && contents != state.debug_output {
            state.debug_output = contents;

            if let Ok(mut file) = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open("debug_output.log")
            {}
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(Visuals::dark());

        check_debug_output(&mut self.state);

        egui::SidePanel::left("project_panel").show(ctx, |ui| {
            ui.set_width(200.0);
            ui.heading("GeneNet");
            ui.separator();
            ui::side_panel(ui, &mut self.state);
        });

        egui::TopBottomPanel::bottom("debug_panel")
            .resizable(true)
            .min_height(50.0)
            .default_height(self.state.debug_panel_height)
            .show_animated(ctx, self.state.debug_panel_visible, |ui| {
                self.state.debug_panel_height = ui.available_height();

                ui.horizontal(|ui| {
                    ui.heading("Debug Output");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear").clicked() {
                            self.state.debug_output.clear();
                        }
                        if ui.button("Hide").clicked() {
                            self.state.debug_panel_visible = false;
                        }
                    });
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.state.debug_output)
                                .desired_width(f32::INFINITY)
                                .desired_rows(10)
                                .font(egui::TextStyle::Monospace)
                                .code_editor()
                                .lock_focus(false)
                                .interactive(false),
                        );
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::tab_bar(ui, &mut self.state);
            ui.separator();

            match self.state.selected_tab {
                Tab::Preview => ui::preview_tab(ctx, ui, &mut self.state),
                Tab::BarChart => ui::bar_chart_tab(ctx, ui, &mut self.state),
                Tab::Subgraph => ui::subgraph_tab(ctx, ui, &mut self.state),
                Tab::TopGenesGraph => ui::top_genes_graph_tab(ctx, ui, &mut self.state),
                Tab::Predict => ui::predict_tab(ctx, ui, &mut self.state),
            }

            if !self.state.debug_panel_visible {
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    if ui.button("Show Debug Panel").clicked() {
                        self.state.debug_panel_visible = true;
                    }
                });
            }
        });

        if let Some((path, note_idx)) = &mut self.state.note_editing {
            if let Some(proj) = get_project_mut(&mut self.state.projects, path) {
                if let Some(note) = proj.notes.get_mut(*note_idx) {
                    let window_id = egui::Id::new((path.clone(), *note_idx));
                    egui::Window::new(&note.name)
                        .id(window_id)
                        .resizable(true)
                        .collapsible(false)
                        .default_size([400.0, 300.0])
                        .show(ctx, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Note Name:");
                                ui.text_edit_singleline(&mut note.name);
                            });

                            ui.separator();

                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.add(
                                    egui::TextEdit::multiline(&mut note.content)
                                        .desired_rows(20)
                                        .desired_width(f32::INFINITY)
                                        .code_editor(),
                                );
                            });

                            ui.separator();
                            if ui.button("Close").clicked() {
                                self.state.note_editing = None;
                            }
                        });
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "GeneNet",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )?;

    Ok(())
}
