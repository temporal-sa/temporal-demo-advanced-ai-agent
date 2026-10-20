// Syntax coloring for fenced code in an agent's reply. The core build plus the languages a
// coding session actually emits, so an untagged fence can still be guessed (a Go file with no
// ```go marker, for instance) without shipping every grammar.

import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import csharp from "highlight.js/lib/languages/csharp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import go from "highlight.js/lib/languages/go";
import ini from "highlight.js/lib/languages/ini";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import kotlin from "highlight.js/lib/languages/kotlin";
import markdown from "highlight.js/lib/languages/markdown";
import php from "highlight.js/lib/languages/php";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import ruby from "highlight.js/lib/languages/ruby";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("python", python);
hljs.registerLanguage("go", go);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml);
hljs.registerLanguage("rust", rust);
hljs.registerLanguage("java", java);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("diff", diff);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("css", css);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("ruby", ruby);
hljs.registerLanguage("c", c);
hljs.registerLanguage("cpp", cpp);
hljs.registerLanguage("csharp", csharp);
hljs.registerLanguage("kotlin", kotlin);
hljs.registerLanguage("php", php);
hljs.registerLanguage("ini", ini);
hljs.registerLanguage("dockerfile", dockerfile);
hljs.registerLanguage("plaintext", plaintext);

// Fence tags highlight.js doesn't already alias to a registered grammar.
const EXTRA: Record<string, string> = {
  docker: "dockerfile",
  golang: "go",
  zsh: "bash",
  console: "bash",
  text: "plaintext",
  txt: "plaintext",
};

// Below this, a guess is too weak (prose in a fence scores about 2) and the block stays plain.
const MIN_RELEVANCE = 3;

function escapeHtml(source: string): string {
  return source.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** The grammar name for a fence's info string (`go`, `main.go`, `docker`), if we have one. */
export function languageName(info: string | undefined): string | undefined {
  if (!info) return;
  const token = info.trim().split(/\s+/)[0];
  if (!token) return;
  if (hljs.getLanguage(token)) return token;
  const mapped = EXTRA[token.toLowerCase()];
  if (mapped && hljs.getLanguage(mapped)) return mapped;
  const ext = token.includes(".") ? token.slice(token.lastIndexOf(".") + 1).toLowerCase() : "";
  if (!ext) return;
  if (hljs.getLanguage(ext)) return ext;
  const fromExt = EXTRA[ext];
  if (fromExt && hljs.getLanguage(fromExt)) return fromExt;
  return;
}

/** Highlighted HTML for a fence. Escaped plain text when the language is unknown or the guess
 *  is too weak to trust. Safe to inject: highlight.js escapes the source. */
export function highlightCode(source: string, info?: string): string {
  const lang = languageName(info);
  try {
    if (lang === "plaintext") return escapeHtml(source);
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(source, { language: lang, ignoreIllegals: true }).value;
    }
    const auto = hljs.highlightAuto(source);
    if (auto.language && auto.relevance >= MIN_RELEVANCE) return auto.value;
  } catch {
    // A fence still being typed can trip a grammar; show the text rather than drop the block.
  }
  return escapeHtml(source);
}
