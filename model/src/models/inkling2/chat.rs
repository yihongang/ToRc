//! chat.rs - Inkling 2's chat format, from the chat template the model ships with: the thinking-effort system turn,
//! the tool declaration, the message/content tokens, and the `content_invoke_tool_json` call format.
//!
//! The template writes each turn as `<|message_user|>` / `<|message_model|>` / `<|message_system|>` /
//! `<|message_tool|>`, then `<|content_text|>`, `<|content_thinking|>` or `<|content_invoke_tool_json|>`, and closes
//! it with `<|end_message|>`. A model turn also ends with `<|content_model_end_sampling|>`.

use serde_json::{Map, Value};

/// The markers the format writes that a vocabulary may keep as added tokens rather than control tokens.
pub const MARKERS: &[&str] = &[
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_model_end_sampling|>",
    "<|content_invoke_tool_json|>",
    "<|content_xml|>",
    "<|content_image|>",
    "<|end_message|>",
    "<|unused_200054|>",
];

pub const END: &str = "<|content_model_end_sampling|>";

/// The effort names the template's `effort_map`, and its default (0.9).
fn effort(name: Option<&str>, think: Option<bool>) -> f32 {
    match name {
        Some("none") => 0.0,
        Some("minimal") => 0.1,
        Some("low") => 0.2,
        Some("medium") => 0.7,
        Some("high") | Some("xhigh") | Some("max") => 0.99,
        None if think == Some(false) => 0.0,
        _ => 0.9,
    }
}

fn effort_turn(e: f32) -> String {
    let n = if e == 0.0 { "0".to_string() } else { e.to_string() };
    format!("<|message_system|><|content_text|>Thinking effort level: {n}<|end_message|>")
}

/// JSON with every object's keys sorted, compact (the template's `tojson(sort_keys=true)`).
fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut ks: Vec<&String> = m.keys().collect();
            ks.sort();
            let mut o = Map::new();
            for k in ks {
                o.insert(k.clone(), sorted(&m[k]));
            }
            Value::Object(o)
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        v => v.clone(),
    }
}

fn tool_specs(tools: &[Value]) -> String {
    let specs: Vec<Value> = tools
        .iter()
        .map(|t| {
            let f = if t["function"].is_object() { &t["function"] } else { t };
            let mut o = Map::new();
            o.insert(
                "description".into(),
                if f["description"].is_string() { f["description"].clone() } else { Value::String(String::new()) },
            );
            o.insert("name".into(), f["name"].clone());
            o.insert(
                "parameters".into(),
                if f["parameters"].is_object() { f["parameters"].clone() } else { Value::Object(Map::new()) },
            );
            o.insert(
                "type".into(),
                if t["type"].is_null() { Value::String("function".into()) } else { t["type"].clone() },
            );
            Value::Object(o)
        })
        .collect();
    serde_json::to_string(&sorted(&Value::Array(specs))).unwrap_or_default()
}

fn role_token(role: &str) -> &'static str {
    match role {
        "assistant" => "<|message_model|>",
        "system" => "<|message_system|>",
        "tool" => "<|message_tool|>",
        _ => "<|message_user|>",
    }
}

fn name_of(m: &Value) -> String {
    m["name"].as_str().unwrap_or("").to_string()
}

/// A call as the template writes it: `{"name":..., "args":{...}}`, sorted and compact.
fn call_text(name: &str, args: &Map<String, Value>) -> String {
    let mut o = Map::new();
    o.insert("name".into(), Value::String(name.into()));
    o.insert("args".into(), Value::Object(args.clone()));
    serde_json::to_string(&sorted(&Value::Object(o))).unwrap_or_default()
}

/// The messages in Inkling 2's format. `effort_name` is the request's reasoning effort; `think` false means the
/// template's 0.0 level.
pub fn prompt(msgs: &[Value], tools: &[Value], effort_name: Option<&str>, think: Option<bool>) -> String {
    let mut s = String::new();
    if !tools.is_empty() {
        s += &format!("<|message_system|>tool_declare<|content_xml|>{}<|end_message|>", tool_specs(tools));
    }
    let e = effort(effort_name, think);
    let mut emitted = false;
    let mut call_names: std::collections::HashMap<String, String> = Default::default();
    for m in msgs {
        let role = m["role"].as_str().unwrap_or("user");
        if !emitted && role != "system" {
            s += &effort_turn(e);
            emitted = true;
        }
        let mut head = role_token(role).to_string();
        // a tool answer names the tool it answers; an assistant turn may name itself
        if role == "assistant" {
            head += &name_of(m);
        } else if role == "tool" {
            let mut name = name_of(m);
            if name.is_empty()
                && let Some(id) = m["tool_call_id"].as_str()
            {
                name = call_names.get(id).cloned().unwrap_or_default();
            }
            head += &name;
        }
        let calls = m["tool_calls"].as_array().filter(|a| !a.is_empty());
        let reasoning = m["reasoning_content"].as_str().filter(|r| !r.is_empty());
        if role == "assistant"
            && let Some(r) = reasoning
        {
            s += &format!("{head}<|content_thinking|>{r}<|end_message|>");
        }
        // the template writes an empty content turn for a user or system message, and for an assistant turn with
        // neither text, reasoning nor calls
        let empty_ok = role != "assistant" || (calls.is_none() && reasoning.is_none());
        match &m["content"] {
            Value::String(t) if !t.is_empty() || empty_ok => {
                s += &format!("{head}<|content_text|>{t}<|end_message|>");
            }
            Value::Array(parts) => {
                for p in parts {
                    if let Some(t) = p["text"].as_str() {
                        s += &format!("{head}<|content_text|>{t}<|end_message|>");
                    }
                }
            }
            _ => {}
        }
        if role == "assistant"
            && let Some(calls) = calls
        {
            for c in calls {
                let name = c["function"]["name"].as_str().unwrap_or("");
                if let Some(id) = c["id"].as_str() {
                    call_names.insert(id.to_string(), name.to_string());
                }
                let args = match &c["function"]["arguments"] {
                    Value::String(a) => serde_json::from_str::<Map<String, Value>>(a).unwrap_or_default(),
                    Value::Object(o) => o.clone(),
                    _ => Map::new(),
                };
                s += &format!(
                    "<|message_model|>{name}<|content_invoke_tool_json|>{}<|end_message|>",
                    call_text(name, &args)
                );
            }
        }
        if role == "assistant" {
            s += END;
        }
    }
    if !emitted {
        s += &effort_turn(e);
    }
    s
}

/// The calls written in a reply, the reply's content, and its reasoning. A call starts after
/// `<|content_invoke_tool_json|>` and ends at `<|end_message|>`; the name is the turn's, or the JSON's own.
pub fn read(text: &str) -> (Vec<crate::serve::Call>, String, String) {
    let mut calls = Vec::new();
    let mut kept = String::new();
    let mut rest = text;
    loop {
        let Some(at) = rest.find("<|content_invoke_tool_json|>") else {
            kept += rest;
            break;
        };
        let before = &rest[..at];
        // the model turn's opening, and its name, is the text before the call: drop the tag and name, keep text
        if let Some(p) = before.rfind("<|message_model|>") {
            kept += &before[..p];
        } else {
            kept += before;
        }
        let after = &rest[at + "<|content_invoke_tool_json|>".len()..];
        let end = after.find("<|end_message|>").unwrap_or(after.len());
        let json = &after[..end];
        if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(json.trim()) {
            let name = o["name"].as_str().unwrap_or("").to_string();
            let args = match &o["args"] {
                Value::Object(m) => m.clone(),
                _ => Map::new(),
            };
            calls.push(crate::serve::Call { name, args });
        }
        rest = after.get(end + "<|end_message|>".len()..).unwrap_or("");
    }
    let (content, reasoning) = split_markers(&kept);
    (calls, content, reasoning)
}

/// Splits a generated turn into its `content_thinking` and `content_text` parts, dropping the control tags.
fn split_markers(s: &str) -> (String, String) {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut rest = s;
    // 0: content, 1: reasoning
    let mut mode = 0u8;
    loop {
        // pick whichever marker comes first
        let think = rest.find("<|content_thinking|>");
        let text = rest.find("<|content_text|>");
        let (at, tag, m) = match (think, text) {
            (Some(a), Some(b)) if a < b => (a, "<|content_thinking|>", 1),
            (Some(a), None) => (a, "<|content_thinking|>", 1),
            (_, Some(b)) => (b, "<|content_text|>", 0),
            _ => break,
        };
        let chunk = &rest[..at];
        if mode == 1 {
            reasoning += chunk
        } else {
            content += chunk
        }
        rest = &rest[at + tag.len()..];
        mode = m;
    }
    if mode == 1 {
        reasoning += rest
    } else {
        content += rest
    }
    // the turn's closing tags and role opener can be left in the tail
    let clean = |x: &str| {
        x.replace("<|message_model|>", "")
            .replace("<|end_message|>", "")
            .replace(END, "")
            .replace("<|content_text|>", "")
            .replace("<|content_thinking|>", "")
    };
    (clean(&content), clean(&reasoning))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn turns_and_calls() {
        let msgs = vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "Weather?"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
            json!({"role": "tool", "name": "get_weather", "content": "17C"}),
            json!({"role": "assistant", "content": "It is 17C."}),
        ];
        let p = prompt(&msgs, &[], Some("low"), None);
        assert!(p.starts_with("<|message_system|><|content_text|>Be brief.<|end_message|>"));
        assert!(p.contains("<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"));
        assert!(p.contains("<|message_model|>get_weather<|content_invoke_tool_json|>{\"args\":{\"city\":\"Paris\"},\"name\":\"get_weather\"}<|end_message|>"));
        assert!(p.contains("<|message_tool|>get_weather<|content_text|>17C<|end_message|>"));
        assert!(p.contains("<|content_model_end_sampling|>"));
        let (c, kept, think) = read(
            "<|message_model|><|content_thinking|>I should call.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"args\":{\"city\":\"Paris\"},\"name\":\"get_weather\"}<|end_message|><|content_model_end_sampling|>",
        );
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].name.as_str(), c[0].args["city"].as_str()), ("get_weather", Some("Paris")));
        assert!(!kept.contains("invoke_tool"));
        assert_eq!(think, "I should call.");
    }
}
