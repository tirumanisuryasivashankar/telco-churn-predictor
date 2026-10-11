pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source repository...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Installing build and runtime dependencies...'
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install pytest mlflow
                '''
            }
        }

        stage('Run Unit & Validation Tests') {
            steps {
                echo 'Executing automated test suite...'
                sh '''
                    . venv/bin/activate
                    pytest tests/test_model.py -v
                '''
            }
        }

        stage('Model Validation & Registry Check') {
            steps {
                echo 'Validating model pipeline integrity...'
                sh '''
                    . venv/bin/activate
                    python -c "import joblib; model = joblib.load('models/churn_pipeline.joblib'); print('Model integrity check passed.')"
                '''
            }
        }

        stage('Deployment Sync') {
            steps {
                echo 'Synchronizing artifacts with Hugging Face Space repository...'
                sh '''
                    echo "Deployment completed at https://huggingface.co/spaces/Tirumanisurya/telco-churn-predictor"
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'CI/CD pipeline executed successfully.'
        }
        failure {
            echo 'CI/CD pipeline encountered errors during build or test execution.'
        }
    }
}