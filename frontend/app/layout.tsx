import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AgentForge — Trace Viewer",
  description: "Live trace viewer for multi-agent LLM runs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
