# Alternative deployment using Elastic Beanstalk Multi-Container Docker Platform
# This is simpler but less flexible than ECS

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "TOVA"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# S3 Bucket for Elastic Beanstalk
resource "aws_s3_bucket" "eb_app_versions" {
  bucket = "${var.project_name}-${var.environment}-eb-app-versions"
}

resource "aws_s3_bucket_versioning" "eb_app_versions" {
  bucket = aws_s3_bucket.eb_app_versions.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "tova" {
  name        = "${var.project_name}-${var.environment}"
  description = "TOVA Topic Modeling Application"
}

# Elastic Beanstalk Application Version
# Note: You'll need to upload your Docker Compose file and create a zip
# This is a placeholder - you'll need to upload versions manually or via CI/CD
resource "aws_elastic_beanstalk_application_version" "tova" {
  name         = "${var.project_name}-${var.environment}-v1"
  application  = aws_elastic_beanstalk_application.tova.name
  description  = "Initial version"
  bucket       = aws_s3_bucket.eb_app_versions.id
  key          = "app-versions/tova-v1.zip"
  
  # This will fail initially - you need to create and upload the zip first
  lifecycle {
    ignore_changes = [key]
  }
}

# IAM Role for Elastic Beanstalk
resource "aws_iam_role" "eb_service" {
  name = "${var.project_name}-${var.environment}-eb-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "elasticbeanstalk.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eb_service" {
  role       = aws_iam_role.eb_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSElasticBeanstalkService"
}

resource "aws_iam_role" "eb_ec2" {
  name = "${var.project_name}-${var.environment}-eb-ec2"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eb_ec2_web_tier" {
  role       = aws_iam_role.eb_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
}

resource "aws_iam_role_policy_attachment" "eb_ec2_worker_tier" {
  role       = aws_iam_role.eb_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWorkerTier"
}

resource "aws_iam_role_policy_attachment" "eb_ec2_multicontainer" {
  role       = aws_iam_role.eb_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkMulticontainerDocker"
}

resource "aws_iam_instance_profile" "eb_ec2" {
  name = "${var.project_name}-${var.environment}-eb-ec2"
  role = aws_iam_role.eb_ec2.name
}

# RDS PostgreSQL (same as ECS module)
module "rds" {
  source = "../modules/rds"
  
  project_name        = var.project_name
  environment         = var.environment
  vpc_id              = var.vpc_id
  private_subnet_ids  = var.private_subnet_ids
  db_instance_class   = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_name             = var.db_name
  db_username         = var.db_username
  db_password         = var.db_password
  allowed_cidr_blocks = [var.vpc_cidr]
}

# Elastic Beanstalk Environment
resource "aws_elastic_beanstalk_environment" "tova" {
  name                = "${var.project_name}-${var.environment}-env"
  application         = aws_elastic_beanstalk_application.tova.name
  solution_stack_name = "64bit Amazon Linux 2 v3.5.0 running Docker"
  tier                = "WebServer"

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.eb_ec2.name
  }

  setting {
    namespace = "aws:autoscaling:asg"
    name      = "MinSize"
    value     = "1"
  }

  setting {
    namespace = "aws:autoscaling:asg"
    name      = "MaxSize"
    value     = "4"
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "LoadBalancerType"
    value     = "application"
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment:process:default"
    name      = "HealthCheckPath"
    value     = "/health"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "DATABASE_URL"
    value     = "postgresql://${var.db_username}:${var.db_password}@${module.rds.db_endpoint}/${var.db_name}"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "POSTGRES_USER"
    value     = var.db_username
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "POSTGRES_PASSWORD"
    value     = var.db_password
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "POSTGRES_DB"
    value     = var.db_name
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "OPENAI_API_KEY"
    value     = var.openai_api_key
  }

  setting {
    namespace = "aws:ec2:vpc"
    name      = "VPCId"
    value     = var.vpc_id
  }

  setting {
    namespace = "aws:ec2:vpc"
    name      = "Subnets"
    value     = join(",", var.private_subnet_ids)
  }

  setting {
    namespace = "aws:ec2:vpc"
    name      = "ELBSubnets"
    value     = join(",", var.public_subnet_ids)
  }

  setting {
    namespace = "aws:elasticbeanstalk:healthreporting:system"
    name      = "SystemType"
    value     = "enhanced"
  }

  # Note: You'll need to create a Dockerrun.aws.json file
  # This file should reference your docker-compose.yaml or define containers directly
}

