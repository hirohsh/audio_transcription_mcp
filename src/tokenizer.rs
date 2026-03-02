use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct WhisperTokenizer {
    tokenizer: Tokenizer,
}

// Whisper special token IDs
pub const SOT_TOKEN: u32 = 50258;
pub const EOT_TOKEN: u32 = 50257;
pub const _TRANSLATE_TOKEN: u32 = 50359;
pub const TRANSCRIBE_TOKEN: u32 = 50360;
pub const NO_TIMESTAMPS_TOKEN: u32 = 50364;

/// Language code to Whisper token ID mapping
pub fn language_token(lang: &str) -> Option<u32> {
    let offset = match lang {
        "en" => 0,
        "zh" => 1,
        "de" => 2,
        "es" => 3,
        "ru" => 4,
        "ko" => 5,
        "fr" => 6,
        "ja" => 7,
        "pt" => 8,
        "tr" => 9,
        "pl" => 10,
        "ca" => 11,
        "nl" => 12,
        "ar" => 13,
        "sv" => 14,
        "it" => 15,
        "id" => 16,
        "hi" => 17,
        "fi" => 18,
        "vi" => 19,
        "he" => 20,
        "uk" => 21,
        "el" => 22,
        "ms" => 23,
        "cs" => 24,
        "ro" => 25,
        "da" => 26,
        "hu" => 27,
        "ta" => 28,
        "no" => 29,
        "th" => 30,
        "ur" => 31,
        "hr" => 32,
        "bg" => 33,
        "lt" => 34,
        "la" => 35,
        "mi" => 36,
        "ml" => 37,
        "cy" => 38,
        "sk" => 39,
        "te" => 40,
        "fa" => 41,
        "lv" => 42,
        "bn" => 43,
        "sr" => 44,
        "az" => 45,
        "sl" => 46,
        "kn" => 47,
        "et" => 48,
        "mk" => 49,
        "br" => 50,
        "eu" => 51,
        "is" => 52,
        "hy" => 53,
        "ne" => 54,
        "mn" => 55,
        "bs" => 56,
        "kk" => 57,
        "sq" => 58,
        "sw" => 59,
        "gl" => 60,
        "mr" => 61,
        "pa" => 62,
        "si" => 63,
        "km" => 64,
        "sn" => 65,
        "yo" => 66,
        "so" => 67,
        "af" => 68,
        "oc" => 69,
        "ka" => 70,
        "be" => 71,
        "tg" => 72,
        "sd" => 73,
        "gu" => 74,
        "am" => 75,
        "yi" => 76,
        "lo" => 77,
        "uz" => 78,
        "fo" => 79,
        "ht" => 80,
        "ps" => 81,
        "tk" => 82,
        "nn" => 83,
        "mt" => 84,
        "sa" => 85,
        "lb" => 86,
        "my" => 87,
        "bo" => 88,
        "tl" => 89,
        "mg" => 90,
        "as" => 91,
        "tt" => 92,
        "haw" => 93,
        "ln" => 94,
        "ha" => 95,
        "ba" => 96,
        "jw" => 97,
        "su" => 98,
        _ => return None,
    };
    Some(SOT_TOKEN + 1 + offset)
}

impl WhisperTokenizer {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Decode token IDs to text, filtering out special tokens.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| id < SOT_TOKEN)
            .collect();

        let text = self
            .tokenizer
            .decode(&filtered, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;

        Ok(text.trim().to_string())
    }

    /// Build the initial decoder prompt tokens.
    /// Format: <|startoftranscript|> [<|lang|>] <|transcribe|> <|notimestamps|>
    pub fn build_prompt(&self, language: Option<&str>) -> Vec<i64> {
        let mut tokens = vec![SOT_TOKEN as i64];

        if let Some(lang) = language {
            if let Some(lang_token) = language_token(lang) {
                tokens.push(lang_token as i64);
            }
        }

        tokens.push(TRANSCRIBE_TOKEN as i64);
        tokens.push(NO_TIMESTAMPS_TOKEN as i64);
        tokens
    }
}
