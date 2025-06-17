use std::fs::File;
use std::path::Path;
use polars::prelude::*;
use std::error::Error;
use dotenv::dotenv;
use std::env;

pub fn get_infer_schema_length() -> usize {
    dotenv().ok();
    match env::var("INFER_SCHEMA_LENGTH") {
        Ok(val) => val.parse::<usize>().unwrap_or(1_000_000),
        Err(_) => 1_000_000,
    }
}

pub fn load_csv_dataset<P: AsRef<Path>>(
    file_path: P,
    infer_schema_length: Option<usize>,
) -> Result<DataFrame, Box<dyn Error>> {
    let schema_length = infer_schema_length.unwrap_or_else(get_infer_schema_length);
    let df = CsvReader::new(File::open(file_path)?)
        .with_options(CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(schema_length))
        )
        .finish()?
        .with_row_index(PlSmallStr::from("Index"), Some(0))?;

    Ok(df)
}
