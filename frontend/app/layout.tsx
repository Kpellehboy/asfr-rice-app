import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASFR Rice Disease Classification",
  description:
    "Research prototype for image-level rice disease classification with Grad-CAM and ASFR spatial-gate visualizations.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
