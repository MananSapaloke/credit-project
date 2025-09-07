import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { Toaster } from 'react-hot-toast';
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/next';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'CreditLens - AI-Powered Credit Risk Assessment',
  description: 'Get instant credit risk analysis with transparent AI explanations and actionable recommendations. Secure, fast, and reliable credit scoring platform.',
  keywords: 'credit score, risk assessment, AI, machine learning, loan application, financial analysis',
  authors: [{ name: 'CreditLens Team' }],
  creator: 'CreditLens',
  publisher: 'CreditLens',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://creditlens.vercel.app'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'CreditLens - AI-Powered Credit Risk Assessment',
    description: 'Get instant credit risk analysis with transparent AI explanations and actionable recommendations.',
    url: 'https://creditlens.vercel.app',
    siteName: 'CreditLens',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'CreditLens - AI Credit Risk Assessment',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CreditLens - AI-Powered Credit Risk Assessment',
    description: 'Get instant credit risk analysis with transparent AI explanations and actionable recommendations.',
    images: ['/og-image.png'],
    creator: '@creditlens',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} antialiased`}>
        <Providers>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
          <Analytics />
          <SpeedInsights />
        </Providers>
      </body>
    </html>
  );
}
