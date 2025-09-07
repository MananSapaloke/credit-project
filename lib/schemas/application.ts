import { z } from 'zod';

export const ApplicationSchema = z.object({
  // Optional fields
  SK_ID_CURR: z.number().optional(),
  
  // Required personal information
  age_years: z.number().min(18, 'Must be at least 18 years old').max(80, 'Must be under 80 years old'),
  employment_years: z.number().min(0, 'Employment years cannot be negative').max(50, 'Employment years cannot exceed 50'),
  
  // Required financial information
  AMT_INCOME_TOTAL: z.number().min(1, 'Annual income must be greater than 0'),
  AMT_CREDIT: z.number().min(1, 'Loan amount must be greater than 0'),
  
  // Optional financial information
  AMT_ANNUITY: z.number().min(0, 'Annuity cannot be negative').optional(),
  AMT_GOODS_PRICE: z.number().min(0, 'Goods price cannot be negative').optional(),
  CNT_CHILDREN: z.number().min(0, 'Number of children cannot be negative').max(10, 'Number of children cannot exceed 10'),
  CNT_FAM_MEMBERS: z.number().min(1, 'Family size must be at least 1').max(10, 'Family size cannot exceed 10'),
  
  // Asset ownership
  FLAG_OWN_CAR: z.enum(['Yes', 'No']),
  FLAG_OWN_REALTY: z.enum(['Yes', 'No']),
  
  // Demographics
  NAME_INCOME_TYPE: z.enum([
    'Businessman',
    'Commercial associate',
    'Maternity leave',
    'Pensioner',
    'State servant',
    'Student',
    'Unemployed',
    'Working'
  ]),
  NAME_EDUCATION_TYPE: z.enum([
    'Academic degree',
    'Higher education',
    'Incomplete higher',
    'Lower secondary',
    'Secondary / secondary special'
  ]),
  NAME_FAMILY_STATUS: z.enum([
    'Civil marriage',
    'Married',
    'Separated',
    'Single / not married',
    'Unknown',
    'Widow'
  ]),
  OCCUPATION_TYPE: z.enum([
    'Accountants',
    'Cleaning staff',
    'Cooking staff',
    'Core staff',
    'Drivers',
    'HR staff',
    'High skill tech staff',
    'IT staff',
    'Laborers',
    'Low-skill Laborers',
    'Managers',
    'Medicine staff',
    'Private service staff',
    'Realty agents',
    'Sales staff',
    'Secretaries',
    'Security staff',
    'Waiters/barmen staff'
  ]),
  NAME_HOUSING_TYPE: z.enum([
    'Co-op apartment',
    'House / apartment',
    'Municipal apartment',
    'Office apartment',
    'Rented apartment',
    'With parents'
  ]),
  
  // Additional optional fields
  previous_defaults: z.number().min(0, 'Previous defaults cannot be negative'),
  bureau_overdue_amount: z.number().min(0, 'Bureau overdue amount cannot be negative'),
  pos_dpd_max: z.number().min(0, 'POS DPD cannot be negative'),
}).refine((data) => {
  // Business rule: Loan amount cannot exceed 10x annual income
  return data.AMT_CREDIT <= data.AMT_INCOME_TOTAL * 10;
}, {
  message: 'Loan amount cannot exceed 10 times your annual income',
  path: ['AMT_CREDIT'],
});

export type ApplicationData = z.infer<typeof ApplicationSchema>;
