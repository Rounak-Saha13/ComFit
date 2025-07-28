import { useEffect, useRef } from "react";
import renderMathInElement from "katex/contrib/auto-render";
import "katex/dist/katex.min.css";

interface Props {
  html: string;
  className?: string;
}

// Converts only [ ... ] blocks that begin with a LaTeX backslash (e.g., [ \frac{a}{b} ])
function convertLatexBrackets(input: string) {
  return input.replace(
    /\[\s*\\(.+?)\s*\]/g,
    (_, expr) => `$$\\${expr.trim()}$$`
  );
}

export default function FormattedContent({ html, className = "" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      const convertedHtml = convertLatexBrackets(html);
      containerRef.current.innerHTML = convertedHtml;

      // Render math content
      renderMathInElement(containerRef.current, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        fleqn: false,
        output: "html",
      });
    }
  }, [html]);

  return <div ref={containerRef} className={className} />;
}
