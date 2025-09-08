"use client";

import React from "react";

export const FloatingActionButton: React.FC<{ onClick?: () => void }> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="fixed bottom-6 right-6 h-12 px-4 bg-black text-white rounded-full shadow"
    >
      Get Started
    </button>
  );
};


