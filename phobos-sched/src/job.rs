use anyhow::{Context, Result, bail};
use phobos_cluster::proto::{self, DimensionBinding, Job, ScalarBinding, TensorInput};
use phobos_cluster::tile::{AccessMode, DataType, ScalarValue};

pub fn parse_job(path: &str) -> Result<Job> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading job-file {path}"))?;
    let mut source = None;
    let mut dimensions = Vec::new();
    let mut tensors = Vec::new();
    let mut scalars = Vec::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let err = |msg: &str| anyhow::anyhow!("{path}:{}: {msg}", lineno + 1);

        if let Some(rest) = line.strip_prefix("source") {
            let p = rest
                .trim_start()
                .strip_prefix('=')
                .ok_or_else(|| err("expected `source = <path>`"))?;
            let src = std::fs::read_to_string(p.trim()).with_context(|| {
                format!("{path}:{}: reading source file '{}'", lineno + 1, p.trim())
            })?;
            source = Some(src);
        } else if let Some(rest) = line.strip_prefix("dim ") {
            // dim NAME = VALUE
            let (name, value) = rest
                .split_once('=')
                .ok_or_else(|| err("expected `dim NAME = VALUE`"))?;
            dimensions.push(DimensionBinding {
                name: name.trim().to_string(),
                value: value
                    .trim()
                    .parse()
                    .map_err(|_| err("dim value must be an integer"))?,
            });
        } else if let Some(rest) = line.strip_prefix("tensor ") {
            // tensor NAME = MODE DATA_TYPE D0xD1x... URI
            let (name, spec) = rest
                .split_once('=')
                .ok_or_else(|| err("expected `tensor NAME = MODE DATA_TYPE SHAPE URI`"))?;
            let mut parts = spec.split_whitespace();
            let mode = parts.next().ok_or_else(|| err("tensor missing mode"))?;
            let data_type = parts
                .next()
                .ok_or_else(|| err("tensor missing data type"))?;
            let shape = parts.next().ok_or_else(|| err("tensor missing shape"))?;
            let uri = parts.next().ok_or_else(|| err("tensor missing uri"))?;
            tensors.push(TensorInput {
                name: name.trim().to_string(),
                data_type: proto::data_type_to_i32(
                    parse_data_type(data_type).map_err(|m| err(&m))?,
                ),
                shape: phobos_base::shape::parse(shape).map_err(|e| err(&e.to_string()))?,
                mode: proto::am_to_i32(parse_mode(mode).map_err(|m| err(&m))?),
                uri: uri.to_string(),
            });
        } else if let Some(rest) = line.strip_prefix("scalar ") {
            // scalar NAME = DATA_TYPE VALUE
            let (name, spec) = rest
                .split_once('=')
                .ok_or_else(|| err("expected `scalar NAME = DATA_TYPE VALUE`"))?;
            let mut parts = spec.split_whitespace();
            let data_type = parts
                .next()
                .ok_or_else(|| err("scalar missing data type"))?;
            let value = parts.next().ok_or_else(|| err("scalar missing value"))?;
            scalars.push(ScalarBinding {
                name: name.trim().to_string(),
                value: Some(proto::scalar_value_to_proto(
                    parse_scalar(data_type, value).map_err(|m| err(&m))?,
                )),
            });
        } else {
            bail!("{path}:{}: unknown directive '{line}'", lineno + 1);
        }
    }

    Ok(Job {
        source: source.context("job-file has no `source =` line")?,
        dimensions,
        tensors,
        scalars,
    })
}

fn parse_scalar(data_type: &str, value: &str) -> std::result::Result<ScalarValue, String> {
    let bad = |kind: &str| format!("scalar value '{value}' is not a valid {kind}");
    match data_type {
        "f32" => Ok(ScalarValue::F32(value.parse().map_err(|_| bad("f32"))?)),
        "f64" => Ok(ScalarValue::F64(value.parse().map_err(|_| bad("f64"))?)),
        "i32" => Ok(ScalarValue::I32(value.parse().map_err(|_| bad("i32"))?)),
        "i64" => Ok(ScalarValue::I64(value.parse().map_err(|_| bad("i64"))?)),
        "bool" => Ok(ScalarValue::Bool(value.parse().map_err(|_| bad("bool"))?)),
        _ => Err(format!("unknown scalar data type '{data_type}'")),
    }
}

fn parse_mode(s: &str) -> std::result::Result<AccessMode, String> {
    match s {
        "read" => Ok(AccessMode::Read),
        "write" => Ok(AccessMode::Write),
        "rmw" => Ok(AccessMode::RMW),
        _ => Err(format!("unknown tensor mode '{s}' (read|write|rmw)")),
    }
}

fn parse_data_type(s: &str) -> std::result::Result<DataType, String> {
    match s {
        "f16" => Ok(DataType::F16),
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "i8" => Ok(DataType::I8),
        "i32" => Ok(DataType::I32),
        "i64" => Ok(DataType::I64),
        "bool" => Ok(DataType::Bool),
        _ => Err(format!("unknown data type '{s}'")),
    }
}

