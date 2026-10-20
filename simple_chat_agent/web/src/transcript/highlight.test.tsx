import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Markdown } from "./Markdown";
import { highlightCode, languageName } from "./highlight";

const GO = `package main

import (
\t"context"
)

func main() {
\tlog.Fatalln("unable to create client", err)
}
`;

describe("fenced code is highlighted", () => {
  it("colors a Go fence, including one that names the file instead of the language", () => {
    expect(languageName("go")).toBe("go");
    expect(languageName("starter/main.go")).toBe("go");
    const html = highlightCode(GO, "go");
    expect(html).toContain('class="hljs-keyword"');
    expect(html).toContain("package");
    expect(html).toContain("hljs-string");
    expect(html).not.toContain("<script");
  });

  it("guesses Go when the fence has no language", () => {
    expect(highlightCode(GO)).toContain("hljs-keyword");
  });

  it("leaves prose and inline code uncolored, and escapes markup in a fence", () => {
    expect(highlightCode("This is just a note about the refund service.\n")).not.toContain("hljs-");
    const fenced = renderToStaticMarkup(<Markdown text={"See `package` and then:\n\n```go\npackage main\nfunc main() {}\n<script>alert(1)</script>\n```\n"} />);
    expect(fenced).toContain("<code>package</code>");
    expect(fenced).toContain("hljs-keyword");
    expect(fenced).not.toContain("<script>");
  });
});
