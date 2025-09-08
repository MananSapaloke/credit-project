"use client";

import React from "react";
import Link from "next/link";

export const Header: React.FC = () => {
  return (
    <header className="w-full border-b">
      <div className="container mx-auto px-4 h-14 flex items-center justify-between">
        <Link href="/" className="font-semibold">CreditLens</Link>
        <nav className="space-x-4">
          <Link href="/application">Application</Link>
          <Link href="/results">Results</Link>
          <Link href="/admin">Admin</Link>
        </nav>
      </div>
    </header>
  );
};


