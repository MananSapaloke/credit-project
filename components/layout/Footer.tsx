"use client";

import React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t mt-12">
      <div className="container mx-auto px-4 h-14 flex items-center justify-between text-sm text-gray-500">
        <span>© {new Date().getFullYear()} CreditLens</span>
      </div>
    </footer>
  );
};


