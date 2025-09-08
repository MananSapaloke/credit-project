"use client";

import React from "react";

export const CookieConsent: React.FC<{ onAccept?: () => void }> = ({ onAccept }) => {
  return (
    <div className="fixed bottom-0 inset-x-0 bg-white border-t p-4 flex items-center justify-between">
      <p className="text-sm">We use cookies to improve your experience.</p>
      <button className="px-3 py-1 bg-black text-white rounded" onClick={onAccept}>Accept</button>
    </div>
  );
};


