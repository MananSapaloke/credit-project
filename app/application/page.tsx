'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import toast from 'react-hot-toast';

// Components
import { Header } from '@/components/layout/Header';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { Step1PersonalInfo } from '@/components/application/Step1PersonalInfo';
import { Step2FinancialInfo } from '@/components/application/Step2FinancialInfo';
import { Step3AdditionalInfo } from '@/components/application/Step3AdditionalInfo';
import { Step4Review } from '@/components/application/Step4Review';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ApplicationSchema } from '@/lib/schemas/application';

type ApplicationData = z.infer<typeof ApplicationSchema>;

const steps = [
  { id: 1, title: 'Personal Information', description: 'Basic details about yourself' },
  { id: 2, title: 'Financial Information', description: 'Income and loan details' },
  { id: 3, title: 'Additional Information', description: 'Assets and other details' },
  { id: 4, title: 'Review & Submit', description: 'Review your application' },
];

export default function ApplicationPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState<Partial<ApplicationData>>({});
  const router = useRouter();

  const methods = useForm<ApplicationData>({
    resolver: zodResolver(ApplicationSchema),
    mode: 'onChange',
    defaultValues: {
      AMT_INCOME_TOTAL: 0,
      AMT_CREDIT: 0,
      age_years: 25,
      employment_years: 1,
      CNT_CHILDREN: 0,
      CNT_FAM_MEMBERS: 1,
      FLAG_OWN_CAR: 'No',
      FLAG_OWN_REALTY: 'No',
      NAME_INCOME_TYPE: 'Working',
      NAME_EDUCATION_TYPE: 'Secondary / secondary special',
      NAME_FAMILY_STATUS: 'Single / not married',
      OCCUPATION_TYPE: 'Core staff',
      NAME_HOUSING_TYPE: 'House / apartment',
      previous_defaults: 0,
      bureau_overdue_amount: 0,
      pos_dpd_max: 0,
    },
  });

  const { handleSubmit, watch, formState: { errors, isValid } } = methods;

  const watchedFields = watch();

  useEffect(() => {
    setFormData(watchedFields);
  }, [watchedFields]);

  const nextStep = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const onSubmit = async (data: ApplicationData) => {
    setIsSubmitting(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Store results in localStorage for demo
      const mockResults = {
        eligibility: 'Manual Review',
        pred_prob: 0.42,
        pred_label: 1,
        recommended_interest_rate: 11.5,
        repayment_schedule: {
          term_months: 60,
          monthly_installment: 11668.0,
          total_payment: 700080.0,
          total_interest: 200080.0,
        },
        decision_reasoning: [
          {
            feature: 'credit_to_income',
            impact_pct_pts: 12.1,
            description: 'High credit-to-income ratio increases default risk',
          },
        ],
        actionable_tips: [
          'Lower requested loan to 350000 to reduce risk to 28%',
          'Add a down payment of 20% to reduce rate by up to 1.5%',
          'Provide 12 months of bank statements to improve underwriting confidence',
        ],
        explainability: {
          top_positive: [
            { feature: 'credit_to_income', shap_value: 0.121 },
            { feature: 'annuity_to_income', shap_value: 0.042 },
          ],
          top_negative: [
            { feature: 'employment_years', shap_value: -0.034 },
          ],
          success: true,
        },
        processing_time_ms: 267.34,
      };

      localStorage.setItem('creditResults', JSON.stringify(mockResults));
      localStorage.setItem('applicationData', JSON.stringify(data));
      
      toast.success('Application submitted successfully!');
      router.push('/results');
    } catch (error) {
      toast.error('Failed to submit application. Please try again.');
      console.error('Submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <Step1PersonalInfo />;
      case 2:
        return <Step2FinancialInfo />;
      case 3:
        return <Step3AdditionalInfo />;
      case 4:
        return <Step4Review data={formData} />;
      default:
        return <Step1PersonalInfo />;
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return formData.age_years && formData.employment_years && formData.NAME_INCOME_TYPE;
      case 2:
        return formData.AMT_INCOME_TOTAL && formData.AMT_CREDIT;
      case 3:
        return true; // Optional fields
      case 4:
        return isValid;
      default:
        return false;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Progress Bar */}
        <div className="mb-8">
          <ProgressBar currentStep={currentStep} totalSteps={steps.length} />
          <div className="mt-4 text-center">
            <h2 className="text-2xl font-bold text-gray-900">
              {steps[currentStep - 1].title}
            </h2>
            <p className="text-gray-600 mt-2">
              {steps[currentStep - 1].description}
            </p>
          </div>
        </div>

        {/* Form */}
        <FormProvider {...methods}>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderStep()}
            </motion.div>

            {/* Navigation Buttons */}
            <div className="flex justify-between pt-8 border-t border-gray-200">
              <button
                type="button"
                onClick={prevStep}
                disabled={currentStep === 1}
                className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>

              {currentStep < steps.length ? (
                <button
                  type="button"
                  onClick={nextStep}
                  disabled={!canProceed()}
                  className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Next Step
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!isValid || isSubmitting}
                  className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
                >
                  {isSubmitting ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Submitting...
                    </>
                  ) : (
                    'Submit Application'
                  )}
                </button>
              )}
            </div>
          </form>
        </FormProvider>
      </div>
    </div>
  );
}
