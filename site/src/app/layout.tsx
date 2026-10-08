import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "UCRA — Uncertainty-aware Capacity Reservation for Network Slicing",
  description:
    "Quantile-forecast reservations with a self-evolving risk dial: 0.0% SLA violations at 71.7% utilization on real 5G RAN traces, validated on the CTTC 5G slicing dataset. Interactive kappa lab and drift demo.",
  keywords: [
    "UCRA",
    "network slicing",
    "capacity reservation",
    "quantile forecasting",
    "5G RAN",
    "self-evolving agent",
  ],
  openGraph: {
    title: "UCRA — reserve the network before demand happens",
    description:
      "0.0% violations @ 71.7% utilization on real RAN traces. Interactive lab.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
