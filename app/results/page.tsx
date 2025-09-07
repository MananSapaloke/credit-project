'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  XCircleIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  ChartBarIcon,
  LightBulbIcon,
  CurrencyDollarIcon,
  CalendarIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

// Components
import { Header } from '@/components/layout/Header';
import { EligibilityBadge } from '@/components/results/EligibilityBadge';
import { RiskGauge } from '@/components/results/RiskGauge';
import { RepaymentSchedule } from '@/components/results/RepaymentSchedule';
import { RiskFactors } from '@/components/results/RiskFactors';
import { ActionableTips } from '@/components/results/ActionableTips';
import { ScenarioSimulator } from '@/components/results/ScenarioSimulator';
import { ShareModal } from '@/components/ui/ShareModal';
import { PDFGenerator } from '@/components/ui/PDFGenerator';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

interface CreditResults {
  eligibility: 'Eligible' | 'Manual Review' | 'Unlikely';
  pred_prob: number;
  pred_label: number;
  recommended_interest_rate: number;
  repayment_schedule: {
    term_months: number;
    monthly_installment: number;
    total_payment: number;
    total_interest: number;
  };
  decision_reasoning: Array<{
    feature: string;
    impact_pct_pts: number;
    description: string;
  }>;
  actionable_tips: string[];
  explainability: {
    top_positive: Array<{ feature: string; shap_value: number }>;
    top_negative: Array<{ feature: string; shap_value: number }>;
    success: boolean;
  };
  processing_time_ms: number;
}

export default function ResultsPage() {
  const [results, setResults] = useState<CreditResults | null>(null);
  const [applicationData, setApplicationData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showShareModal, setShowShareModal] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Load results from localStorage (in real app, this would come from API)
    const storedResults = localStorage.getItem('creditResults');
    const storedApplicationData = localStorage.getItem('applicationData');

    if (storedResults && storedApplicationData) {
      setResults(JSON.parse(storedResults));
      setApplicationData(JSON.parse(storedApplicationData));
    } else {
      toast.error('No results found. Please complete the application first.');
      router.push('/application');
    }

    setIsLoading(false);
  }, [router]);

  const handleDownloadPDF = async () => {
    setIsGeneratingPDF(true);
    try {
      // Simulate PDF generation
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success('PDF report downloaded successfully!');
    } catch (error) {
      toast.error('Failed to generate PDF. Please try again.');
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const handleShare = () => {
    setShowShareModal(true);
  };

  const getEligibilityColor = (eligibility: string) => {
    switch (eligibility) {
      case 'Eligible':
        return 'text-success-600 bg-success-100';
      case 'Manual Review':
        return 'text-warning-600 bg-warning-100';
      case 'Unlikely':
        return 'text-danger-600 bg-danger-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getEligibilityIcon = (eligibility: string) => {
    switch (eligibility) {
      case 'Eligible':
        return CheckCircleIcon;
      case 'Manual Review':
        return ExclamationTriangleIcon;
      case 'Unlikely':
        return XCircleIcon;
      default:
        return ExclamationTriangleIcon;
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!results) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">No Results Found</h2>
          <p className="text-gray-600 mb-8">Please complete the application first.</p>
          <button
            onClick={() => router.push('/application')}
            className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Start Application
          </button>
        </div>
      </div>
    );
  }

  const EligibilityIcon = getEligibilityIcon(results.eligibility);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Your Credit Assessment Results
          </h1>
          <p className="text-gray-600">
            Based on your application, here's your personalized credit risk analysis
          </p>
        </motion.div>

        {/* Main Results Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-2xl shadow-soft p-8 mb-8"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Eligibility & Risk */}
            <div className="space-y-6">
              <div className="text-center">
                <EligibilityBadge 
                  eligibility={results.eligibility}
                  probability={results.pred_prob}
                />
              </div>

              <div className="text-center">
                <RiskGauge 
                  probability={results.pred_prob}
                  label="Default Risk Probability"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <ClockIcon className="w-8 h-8 text-primary-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-gray-900">
                    {results.processing_time_ms}ms
                  </div>
                  <div className="text-sm text-gray-600">Processing Time</div>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <ChartBarIcon className="w-8 h-8 text-primary-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-gray-900">
                    {results.explainability.success ? 'Available' : 'N/A'}
                  </div>
                  <div className="text-sm text-gray-600">AI Explanation</div>
                </div>
              </div>
            </div>

            {/* Right Column - Repayment Schedule */}
            <div>
              <RepaymentSchedule 
                schedule={results.repayment_schedule}
                interestRate={results.recommended_interest_rate}
              />
            </div>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex flex-wrap justify-center gap-4 mb-8"
        >
          <button
            onClick={handleDownloadPDF}
            disabled={isGeneratingPDF}
            className="flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
          >
            {isGeneratingPDF ? (
              <LoadingSpinner size="sm" className="mr-2" />
            ) : (
              <ArrowDownTrayIcon className="w-5 h-5 mr-2" />
            )}
            Download PDF Report
          </button>

          <button
            onClick={handleShare}
            className="flex items-center px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <ShareIcon className="w-5 h-5 mr-2" />
            Share Results
          </button>

          <button
            onClick={() => router.push('/application')}
            className="flex items-center px-6 py-3 bg-success-600 text-white rounded-lg hover:bg-success-700 transition-colors"
          >
            <CheckCircleIcon className="w-5 h-5 mr-2" />
            Apply for Loan
          </button>
        </motion.div>

        {/* Risk Factors */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <RiskFactors 
            reasoning={results.decision_reasoning}
            explainability={results.explainability}
          />
        </motion.div>

        {/* Actionable Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-8"
        >
          <ActionableTips tips={results.actionable_tips} />
        </motion.div>

        {/* Scenario Simulator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <ScenarioSimulator 
            baseData={applicationData}
            currentResults={results}
          />
        </motion.div>
      </div>

      {/* Share Modal */}
      <AnimatePresence>
        {showShareModal && (
          <ShareModal 
            onClose={() => setShowShareModal(false)}
            results={results}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
