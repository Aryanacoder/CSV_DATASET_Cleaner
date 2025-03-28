# Vercel Deployment Instructions

This document provides step-by-step instructions for deploying the AI-Powered CSV Dataset Cleaning Assistant to Vercel.

## Prerequisites

1. A Vercel account (free tier is sufficient)
2. Git installed on your local machine
3. The Vercel CLI (optional, but recommended)

## Deployment Steps

### Option 1: Deploy using the Vercel Dashboard

1. **Create a GitHub repository**
   - Create a new repository on GitHub
   - Upload the contents of the `build` directory to this repository

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com) and sign in
   - Click "Add New..." and select "Project"
   - Import your GitHub repository
   - Configure the project:
     - Framework Preset: Other
     - Root Directory: ./
     - Build Command: None (leave empty)
     - Output Directory: None (leave empty)
   - Click "Deploy"

3. **Environment Variables (Optional)**
   - If needed, add environment variables in the Vercel project settings

### Option 2: Deploy using the Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Navigate to the build directory**
   ```bash
   cd /path/to/improved_csv_cleaner/build
   ```

4. **Deploy to Vercel**
   ```bash
   vercel
   ```
   - Follow the prompts to configure your project
   - When asked about the framework, select "Other"

5. **Set to Production**
   ```bash
   vercel --prod
   ```

## Post-Deployment

After deployment, Vercel will provide you with a URL for your application. The application should be up and running, ready to process CSV files with voice and text commands.

## Troubleshooting

If you encounter any issues during deployment:

1. **Check Logs**: In the Vercel dashboard, navigate to your project and check the deployment logs
2. **Verify Requirements**: Ensure all dependencies in requirements.txt are compatible with Vercel
3. **Check Configuration**: Verify that vercel.json and vercel.py are correctly configured

## Limitations on Vercel

Please note that Streamlit applications on Vercel have some limitations:

1. **File Upload Size**: Limited to the settings in config.toml (currently 1000MB)
2. **Session Duration**: Vercel functions have execution time limits
3. **Statelessness**: Vercel is designed for stateless applications, so session state may not persist between page refreshes

For production use with large datasets or heavy usage, consider deploying to a more robust platform like Heroku, AWS, or Google Cloud.
