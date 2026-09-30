//! `json.dumps(value, ensure_ascii=False)` byte-for-byte, so JSON states tokenize
//! exactly as in the Python runtime (separators, float repr, escaping, key order).
//! Requires serde_json's `preserve_order` and `arbitrary_precision` features.
use serde_json::Value;
use std::fmt::Write;

pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, &n.to_string()),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// Python's json module parses literals without '.', 'e' or 'E' as int, everything else as float.
fn write_number(out: &mut String, text: &str) {
    if text.contains(['.', 'e', 'E']) {
        out.push_str(&float_repr(text.parse::<f64>().unwrap_or(f64::NAN)));
    } else if text == "-0" {
        out.push('0');
    } else {
        out.push_str(text);
    }
}

/// `float.__repr__` (shortest round-trip digits, Python's fixed/scientific switch).
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", value.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("scientific float format");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().expect("float exponent");
    let point = exponent + 1; // value = 0.DIGITS × 10^point
    let mut out = String::with_capacity(digits.len() + 8);
    if value < 0.0 {
        out.push('-');
    }
    let n = digits.len() as i32;
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point >= n {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (point - n) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = point - 1;
        let _ = write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    out
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_match_python_repr() {
        for (v, s) in [
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (32.5, "32.5"),
            (0.1, "0.1"),
            (-2.0, "-2.0"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(float_repr(v), s);
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v: Value = serde_json::from_str(
            r#"{"b": [1, 2.50, -0, 1E5, "é\n\u0001\"\\"], "a": {}, "c": [], "d": null, "e": true}"#,
        )
        .unwrap();
        assert_eq!(
            dumps(&v),
            r#"{"b": [1, 2.5, 0, 100000.0, "é\n\u0001\"\\"], "a": {}, "c": [], "d": null, "e": true}"#
        );
    }
}
