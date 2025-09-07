'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  UsersIcon, 
  ClockIcon, 
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface MetricsData {
  totalApplications: number;
  totalUsers: number;
  avgResponseTime: number;
  modelAccuracy: number;
  dailyApplications: Array<{ date: string; count: number }>;
  eligibilityDistribution: Array<{ name: string; value: number; color: string }>;
  riskDistribution: Array<{ range: string; count: number }>;
  recentApplications: Array<{
    id: string;
    user: string;
    eligibility: string;
    probability: number;
    timestamp: string;
  }>;
}

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    const fetchMetrics = async () => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setMetrics({
        totalApplications: 12543,
        totalUsers: 8921,
        avgResponseTime: 245,
        modelAccuracy: 94.2,
        dailyApplications: [
          { date: '2024-01-01', count: 45 },
          { date: '2024-01-02', count: 52 },
          { date: '2024-01-03', count: 38 },
          { date: '2024-01-04', count: 61 },
          { date: '2024-01-05', count: 47 },
          { date: '2024-01-06', count: 55 },
          { date: '2024-01-07', count: 43 },
        ],
        eligibilityDistribution: [
          { name: 'Eligible', value: 45, color: '#22c55e' },
          { name: 'Manual Review', value: 35, color: '#f59e0b' },
          { name: 'Unlikely', value: 20, color: '#ef4444' },
        ],
        riskDistribution: [
          { range: '0-20%', count: 45 },
          { range: '20-40%', count: 30 },
          { range: '40-60%', count: 15 },
          { range: '60-80%', count: 7 },
          { range: '80-100%', count: 3 },
        ],
        recentApplications: [
          {
            id: 'APP-001',
            user: 'john.doe@email.com',
            eligibility: 'Eligible',
            probability: 0.15,
            timestamp: '2024-01-07 14:30:00',
          },
          {
            id: 'APP-002',
            user: 'jane.smith@email.com',
            eligibility: 'Manual Review',
            probability: 0.42,
            timestamp: '2024-01-07 14:25:00',
          },
          {
            id: 'APP-003',
            user: 'bob.wilson@email.com',
            eligibility: 'Unlikely',
            probability: 0.78,
            timestamp: '2024-01-07 14:20:00',
          },
        ],
      });
      
      setIsLoading(false);
    };

    fetchMetrics();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!metrics) return null;

  const statCards = [
    {
      title: 'Total Applications',
      value: metrics.totalApplications.toLocaleString(),
      icon: ChartBarIcon,
      change: '+12.5%',
      changeType: 'positive' as const,
    },
    {
      title: 'Active Users',
      value: metrics.totalUsers.toLocaleString(),
      icon: UsersIcon,
      change: '+8.2%',
      changeType: 'positive' as const,
    },
    {
      title: 'Avg Response Time',
      value: `${metrics.avgResponseTime}ms`,
      icon: ClockIcon,
      change: '-5.3%',
      changeType: 'positive' as const,
    },
    {
      title: 'Model Accuracy',
      value: `${metrics.modelAccuracy}%`,
      icon: CheckCircleIcon,
      change: '+0.8%',
      changeType: 'positive' as const,
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">Overview of your credit assessment platform</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-xl shadow-soft p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">{stat.value}</p>
                <div className="flex items-center mt-2">
                  {stat.changeType === 'positive' ? (
                    <ArrowTrendingUpIcon className="w-4 h-4 text-success-500 mr-1" />
                  ) : (
                    <ArrowTrendingDownIcon className="w-4 h-4 text-danger-500 mr-1" />
                  )}
                  <span className={`text-sm font-medium ${
                    stat.changeType === 'positive' ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    {stat.change}
                  </span>
                  <span className="text-sm text-gray-500 ml-1">vs last month</span>
                </div>
              </div>
              <div className="p-3 bg-primary-100 rounded-lg">
                <stat.icon className="w-6 h-6 text-primary-600" />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Daily Applications Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-xl shadow-soft p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily Applications</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics.dailyApplications}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Eligibility Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-xl shadow-soft p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Eligibility Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={metrics.eligibilityDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {metrics.eligibilityDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Risk Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-xl shadow-soft p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={metrics.riskDistribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Recent Applications */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="bg-white rounded-xl shadow-soft p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Applications</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Application ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  User
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Eligibility
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Probability
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Timestamp
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {metrics.recentApplications.map((app) => (
                <tr key={app.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {app.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {app.user}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      app.eligibility === 'Eligible' 
                        ? 'bg-success-100 text-success-800'
                        : app.eligibility === 'Manual Review'
                        ? 'bg-warning-100 text-warning-800'
                        : 'bg-danger-100 text-danger-800'
                    }`}>
                      {app.eligibility}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {(app.probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {app.timestamp}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
