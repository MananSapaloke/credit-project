"use client";

import React from "react";

export const CTASection: React.FC<{ onGetStarted?: () => void }> = ({ onGetStarted }) => {
  return (
    <section className="py-12">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Get Started</h2>
          <button className="px-4 py-2 bg-black text-white rounded" onClick={onGetStarted}>Apply Now</button>
        </div>
      </div>
    </section>
  );
};


