use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;
use eframe::egui::Vec2;
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Dataset {
    pub name: String,
    pub path: PathBuf,
    #[serde(skip)]
    pub df: Option<DataFrame>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub name: String,
    pub content: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub name: String,
    pub datasets: Vec<Dataset>,
    pub selected_dataset: Option<usize>,
    pub subprojects: Vec<Project>,
    pub notes: Vec<Note>,
}

#[derive(Debug, Clone)]
pub struct SubgraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
}

#[derive(Debug, Clone)]
pub struct SubgraphEdge {
    pub source: String,
    pub target: String,
}

pub struct AppState {
    pub projects: Vec<Project>,
    pub selected_project_path: Vec<usize>,
    pub selected_tab: Tab,
    pub pending_delete: Option<Vec<usize>>,
    pub note_editing: Option<(Vec<usize>, usize)>,
    pub column_to_remove: Option<String>,
    pub column_to_match: Option<String>,
    pub match_value: String,
    pub graph_pan: Vec2,
    pub graph_zoom: f32,
    pub node_positions: Vec<Vec2>,
    pub layout_done: bool,
    pub bar_genes: Vec<String>,
    pub bar_counts: Vec<f64>,
    pub is_fetching: bool,
    pub bar_selected_value: String,
    pub bar_chart_promise: Option<poll_promise::Promise<Result<ResponseModel, String>>>,
    pub subgraph_gene_symbol: String,
    pub is_fetching_subgraph: bool,
    pub subgraph_promise: Option<poll_promise::Promise<Result<ResponseModel, String>>>,
    pub subgraph_nodes: Vec<SubgraphNode>,
    pub subgraph_edges: Vec<SubgraphEdge>,
    pub subgraph_node_positions: HashMap<String, Vec2>,
    pub subgraph_layout_done: bool,
    pub subgraph_pan: Vec2,
    pub subgraph_zoom: f32,
    pub top_genes_count: String,
    pub is_fetching_top_genes: bool,
    pub top_genes_promise: Option<poll_promise::Promise<Result<ResponseModel, String>>>,
    pub top_genes_nodes: Vec<SubgraphNode>,
    pub top_genes_edges: Vec<SubgraphEdge>,
    pub top_genes_node_positions: HashMap<String, Vec2>,
    pub top_genes_layout_done: bool,
    pub top_genes_pan: Vec2,
    pub top_genes_zoom: f32,
    pub predict_gene_symbol: String,
    pub is_fetching_predict: bool,
    pub is_training_model: bool,
    pub predict_promise: Option<poll_promise::Promise<Result<ResponseModel, String>>>,
    pub predict_nodes: Vec<SubgraphNode>,
    pub predict_edges: Vec<SubgraphEdge>,
    pub predict_edge_scores: HashMap<String, f32>,
    pub predict_node_positions: HashMap<String, Vec2>,
    pub predict_layout_done: bool,
    pub predict_pan: Vec2,
    pub predict_zoom: f32,
    pub predict_threshold: f32,
    pub predict_model_trained: bool,
    pub debug_output: String,
    pub debug_panel_height: f32,
    pub debug_panel_visible: bool,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Tab {
    Preview,
    BarChart,
    Subgraph,
    TopGenesGraph,
    Predict,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RequestModel {
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataframe_csv: Option<String>,
}

impl RequestModel {
    pub fn new(action: &str) -> Self {
        RequestModel {
            action: action.to_string(),
            parameters: None,
            dataframe_csv: None,
        }
    }

    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn with_dataframe(mut self, df: &DataFrame) -> Result<Self, Box<dyn Error>> {
        let mut df_clone = df.clone();
        let mut buf = Vec::new();
        CsvWriter::new(&mut buf)
            .include_header(true)
            .finish(&mut df_clone)?;

        self.dataframe_csv = Some(String::from_utf8(buf)?);
        Ok(self)
    }

    pub fn get_dataframe(&self) -> Result<Option<DataFrame>, Box<dyn Error>> {
        match &self.dataframe_csv {
            Some(csv_str) => {
                let cursor = std::io::Cursor::new(csv_str.as_bytes());
                let df = CsvReader::new(cursor)
                    .with_options(CsvReadOptions::default().with_has_header(true))
                    .finish()?;
                Ok(Some(df))
            }
            None => Ok(None),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponseModel {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataframe_csv: Option<String>,
}

impl ResponseModel {
    pub fn with_result(result: serde_json::Value) -> Self {
        ResponseModel {
            result: Some(result),
            error: None,
            dataframe_csv: None,
        }
    }

    pub fn with_error(error: &str) -> Self {
        ResponseModel {
            result: None,
            error: Some(error.to_string()),
            dataframe_csv: None,
        }
    }

    pub fn with_dataframe(mut self, df: &DataFrame) -> Result<Self, Box<dyn Error>> {
        let mut df_clone = df.clone();
        let mut buf = Vec::new();
        CsvWriter::new(&mut buf)
            .include_header(true)
            .finish(&mut df_clone)?;

        self.dataframe_csv = Some(String::from_utf8(buf)?);
        Ok(self)
    }

    pub fn get_dataframe(&self) -> Result<Option<DataFrame>, Box<dyn Error>> {
        match &self.dataframe_csv {
            Some(csv_str) => {
                let cursor = std::io::Cursor::new(csv_str.as_bytes());
                let df = CsvReader::new(cursor)
                    .with_options(CsvReadOptions::default().with_has_header(true))
                    .finish()?;
                Ok(Some(df))
            }
            None => Ok(None),
        }
    }
}
