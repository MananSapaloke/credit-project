// Google Analytics 4 integration
export const GA_TRACKING_ID = process.env.NEXT_PUBLIC_GA_ID;

// Track page views
export const pageview = (url: string) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', GA_TRACKING_ID, {
      page_path: url,
    });
  }
};

// Track custom events
export const event = ({
  action,
  category,
  label,
  value,
}: {
  action: string;
  category: string;
  label?: string;
  value?: number;
}) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
};

// CreditLens specific tracking
export const trackApplicationStart = () => {
  event({
    action: 'application_start',
    category: 'engagement',
    label: 'user_started_application',
  });
};

export const trackApplicationComplete = (step: number) => {
  event({
    action: 'application_step_complete',
    category: 'engagement',
    label: `step_${step}`,
    value: step,
  });
};

export const trackApplicationSubmit = (data: any) => {
  event({
    action: 'application_submit',
    category: 'conversion',
    label: 'application_submitted',
    value: data.AMT_CREDIT,
  });
};

export const trackResultsView = (eligibility: string, probability: number) => {
  event({
    action: 'results_view',
    category: 'engagement',
    label: eligibility,
    value: Math.round(probability * 100),
  });
};

export const trackPDFDownload = () => {
  event({
    action: 'pdf_download',
    category: 'engagement',
    label: 'report_downloaded',
  });
};

export const trackScenarioSimulation = (scenario: string) => {
  event({
    action: 'scenario_simulation',
    category: 'engagement',
    label: scenario,
  });
};

export const trackShareResults = (method: string) => {
  event({
    action: 'share_results',
    category: 'engagement',
    label: method,
  });
};

// User journey tracking
export const trackUserJourney = (step: string, data?: any) => {
  event({
    action: 'user_journey',
    category: 'navigation',
    label: step,
  });
};

// Error tracking
export const trackError = (error: string, context?: string) => {
  event({
    action: 'error',
    category: 'technical',
    label: error,
  });
};

// Performance tracking
export const trackPerformance = (metric: string, value: number) => {
  event({
    action: 'performance',
    category: 'technical',
    label: metric,
    value: value,
  });
};

// Custom dimensions for CreditLens
export const setCustomDimensions = (dimensions: Record<string, any>) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', GA_TRACKING_ID, {
      custom_map: dimensions,
    });
  }
};

// E-commerce tracking for premium features
export const trackPurchase = (transactionId: string, value: number, currency: string = 'USD') => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', 'purchase', {
      transaction_id: transactionId,
      value: value,
      currency: currency,
    });
  }
};

// Conversion funnel tracking
export const trackFunnelStep = (step: string, stepNumber: number) => {
  event({
    action: 'funnel_step',
    category: 'conversion',
    label: step,
    value: stepNumber,
  });
};

// A/B testing tracking
export const trackABTest = (testName: string, variant: string) => {
  event({
    action: 'ab_test',
    category: 'experiment',
    label: `${testName}_${variant}`,
  });
};

// Heatmap and session recording (if using Hotjar or similar)
export const trackHeatmapEvent = (element: string, action: string) => {
  event({
    action: 'heatmap_interaction',
    category: 'user_behavior',
    label: `${element}_${action}`,
  });
};

// Form abandonment tracking
export const trackFormAbandonment = (formName: string, step: number, data: any) => {
  event({
    action: 'form_abandonment',
    category: 'conversion',
    label: `${formName}_step_${step}`,
    value: step,
  });
};

// Time on page tracking
export const trackTimeOnPage = (page: string, timeSpent: number) => {
  event({
    action: 'time_on_page',
    category: 'engagement',
    label: page,
    value: timeSpent,
  });
};

// Scroll depth tracking
export const trackScrollDepth = (page: string, depth: number) => {
  event({
    action: 'scroll_depth',
    category: 'engagement',
    label: page,
    value: depth,
  });
};

// Search tracking
export const trackSearch = (query: string, results: number) => {
  event({
    action: 'search',
    category: 'engagement',
    label: query,
    value: results,
  });
};

// Social sharing tracking
export const trackSocialShare = (platform: string, content: string) => {
  event({
    action: 'social_share',
    category: 'engagement',
    label: `${platform}_${content}`,
  });
};

// Newsletter signup tracking
export const trackNewsletterSignup = (email: string) => {
  event({
    action: 'newsletter_signup',
    category: 'conversion',
    label: 'email_capture',
  });
};

// Feature usage tracking
export const trackFeatureUsage = (feature: string, usage: number = 1) => {
  event({
    action: 'feature_usage',
    category: 'engagement',
    label: feature,
    value: usage,
  });
};

// API performance tracking
export const trackAPIPerformance = (endpoint: string, responseTime: number, status: number) => {
  event({
    action: 'api_performance',
    category: 'technical',
    label: `${endpoint}_${status}`,
    value: responseTime,
  });
};

// Mobile vs Desktop tracking
export const trackDeviceType = (deviceType: 'mobile' | 'desktop' | 'tablet') => {
  event({
    action: 'device_type',
    category: 'technical',
    label: deviceType,
  });
};

// Browser tracking
export const trackBrowser = (browser: string, version: string) => {
  event({
    action: 'browser_info',
    category: 'technical',
    label: `${browser}_${version}`,
  });
};

// Geographic tracking
export const trackLocation = (country: string, region: string) => {
  event({
    action: 'location',
    category: 'demographics',
    label: `${country}_${region}`,
  });
};

// User segment tracking
export const trackUserSegment = (segment: string, value: any) => {
  event({
    action: 'user_segment',
    category: 'demographics',
    label: segment,
    value: value,
  });
};

// Retention tracking
export const trackRetention = (daysSinceFirstVisit: number) => {
  event({
    action: 'retention',
    category: 'engagement',
    label: 'user_retention',
    value: daysSinceFirstVisit,
  });
};

// Cohort tracking
export const trackCohort = (cohort: string, action: string) => {
  event({
    action: 'cohort_action',
    category: 'engagement',
    label: `${cohort}_${action}`,
  });
};

// Declare gtag function for TypeScript
declare global {
  interface Window {
    gtag: (
      command: 'config' | 'event' | 'js',
      targetId: string,
      config?: Record<string, any>
    ) => void;
  }
}
