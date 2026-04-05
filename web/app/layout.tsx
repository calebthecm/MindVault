import type { Metadata } from "next";
import { Bebas_Neue, DM_Sans } from "next/font/google";
import "./globals.css";

const bebasNeue = Bebas_Neue({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-bebas",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
});

export const metadata: Metadata = {
  title: "MindVault",
  description:
    "A local-first second brain. Chat with your AI conversations, notes, and documents — privately, on your machine.",
  openGraph: {
    title: "MindVault",
    description:
      "A local-first second brain. Chat with your AI conversations, notes, and documents — privately, on your machine.",
    url: "https://mndvlt.com",
    siteName: "MindVault",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "MindVault",
    description: "A local-first second brain. Runs entirely on your machine.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${bebasNeue.variable} ${dmSans.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
