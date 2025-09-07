import nodemailer from 'nodemailer';

interface EmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

class EmailService {
  private transporter: nodemailer.Transporter;

  constructor() {
    this.transporter = nodemailer.createTransporter({
      host: process.env.EMAIL_SERVER_HOST,
      port: parseInt(process.env.EMAIL_SERVER_PORT || '587'),
      secure: false,
      auth: {
        user: process.env.EMAIL_SERVER_USER,
        pass: process.env.EMAIL_SERVER_PASSWORD,
      },
    });
  }

  async sendEmail({ to, subject, html, text }: EmailOptions) {
    try {
      const info = await this.transporter.sendMail({
        from: process.env.EMAIL_FROM,
        to,
        subject,
        html,
        text,
      });

      console.log('Email sent:', info.messageId);
      return { success: true, messageId: info.messageId };
    } catch (error) {
      console.error('Email sending failed:', error);
      return { success: false, error: error.message };
    }
  }

  async sendWelcomeEmail(email: string, name: string) {
    const html = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #2563eb;">Welcome to CreditLens!</h1>
        <p>Hi ${name},</p>
        <p>Thank you for signing up with CreditLens. We're excited to help you with your credit assessment needs.</p>
        <p>You can now:</p>
        <ul>
          <li>Get instant credit risk assessments</li>
          <li>Access detailed explanations of your results</li>
          <li>Receive personalized recommendations</li>
          <li>Download comprehensive reports</li>
        </ul>
        <p>If you have any questions, feel free to contact our support team.</p>
        <p>Best regards,<br>The CreditLens Team</p>
      </div>
    `;

    return this.sendEmail({
      to: email,
      subject: 'Welcome to CreditLens - Your Credit Assessment Platform',
      html,
    });
  }

  async sendAssessmentResults(email: string, name: string, results: any) {
    const html = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #2563eb;">Your Credit Assessment Results</h1>
        <p>Hi ${name},</p>
        <p>Your credit assessment has been completed. Here's a summary of your results:</p>
        
        <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
          <h2 style="color: #1f2937; margin-top: 0;">Assessment Summary</h2>
          <p><strong>Eligibility:</strong> ${results.eligibility}</p>
          <p><strong>Risk Probability:</strong> ${(results.pred_prob * 100).toFixed(1)}%</p>
          <p><strong>Recommended Interest Rate:</strong> ${results.recommended_interest_rate}%</p>
          <p><strong>Monthly Payment:</strong> $${results.repayment_schedule.monthly_installment.toLocaleString()}</p>
        </div>

        <h3>Key Recommendations:</h3>
        <ul>
          ${results.actionable_tips.map((tip: string) => `<li>${tip}</li>`).join('')}
        </ul>

        <p>For detailed analysis and your complete report, please log in to your CreditLens account.</p>
        
        <div style="text-align: center; margin: 30px 0;">
          <a href="${process.env.NEXTAUTH_URL}/results" 
             style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
            View Full Report
          </a>
        </div>

        <p>Best regards,<br>The CreditLens Team</p>
      </div>
    `;

    return this.sendEmail({
      to: email,
      subject: 'Your CreditLens Assessment Results Are Ready',
      html,
    });
  }

  async sendPasswordResetEmail(email: string, resetToken: string) {
    const resetUrl = `${process.env.NEXTAUTH_URL}/reset-password?token=${resetToken}`;
    
    const html = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #2563eb;">Password Reset Request</h1>
        <p>You requested a password reset for your CreditLens account.</p>
        <p>Click the button below to reset your password:</p>
        
        <div style="text-align: center; margin: 30px 0;">
          <a href="${resetUrl}" 
             style="background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">
            Reset Password
          </a>
        </div>

        <p>If you didn't request this password reset, please ignore this email.</p>
        <p>This link will expire in 1 hour.</p>
        
        <p>Best regards,<br>The CreditLens Team</p>
      </div>
    `;

    return this.sendEmail({
      to: email,
      subject: 'Reset Your CreditLens Password',
      html,
    });
  }
}

export const emailService = new EmailService();
