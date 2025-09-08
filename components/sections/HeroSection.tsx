"use client";

import React from "react";

export const HeroSection: React.FC<{ onGetStarted?: () => void }> = ({ onGetStarted }) => {
  return (
    <section className="py-16 bg-gray-50">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold">CreditLens</h1>
        <p className="mt-2 text-gray-600">Modern credit risk assessment platform.</p>
        <button className="mt-6 px-4 py-2 bg-black text-white rounded" onClick={onGetStarted}>Get Started</button>
      </div>
    </section>
  );
};
