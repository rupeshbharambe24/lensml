import type { Metadata } from "next";

import "uplot/dist/uPlot.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "lensml",
  description: "AI agent that supervises ML training runs in real time.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-neutral-950 text-neutral-100 antialiased">{children}</body>
    </html>
  );
}
