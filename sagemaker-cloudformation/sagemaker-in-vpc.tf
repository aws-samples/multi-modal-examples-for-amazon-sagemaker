provider "aws" {
  region = "us-east-1" 
}

data "aws_partition" "current" {}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  mappings = {
    VpcConfigurations = {
      cidr = {
        Vpc           = "10.0.0.0/16"
        PublicSubnet1 = "10.0.10.0/24"
        PrivateSubnet1 = "10.0.20.0/24"
      }
    }
  }
  stack_name = "sagemaker-in-vpc"
}

variable "sage_maker_domain_name" {
  description = "Name of the Studio Domain to Create"
  type        = string
  default     = "SageMaker-GenAI-Domain"
}

resource "aws_s3_bucket" "studio_bucket" {
  bucket = "sagemaker-${data.aws_region.current.name}-${data.aws_caller_identity.current.account_id}"
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["POST", "PUT", "GET", "HEAD", "DELETE"]
    allowed_origins = ["https://*.sagemaker.aws"]
    expose_headers  = ["ETag", "x-amz-delete-marker", "x-amz-id-2", "x-amz-request-id", "x-amz-server-side-encryption", "x-amz-version-id"]
  }
}

resource "aws_vpc" "vpc" {
  cidr_block           = local.mappings["VpcConfigurations"]["cidr"]["Vpc"]
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    "for-use-with-amazon-emr-managed-policies" = "true"
    Name                                        = "${local.stack_name}-VPC"
  }
}

resource "aws_internet_gateway" "internet_gateway" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "${local.stack_name}-IGW"
  }
}

resource "aws_subnet" "public_subnet1" {
  vpc_id                  = aws_vpc.vpc.id
  availability_zone       = element(data.aws_availability_zones.available.names, 0)
  cidr_block              = local.mappings["VpcConfigurations"]["cidr"]["PublicSubnet1"]
  map_public_ip_on_launch = true
  tags = {
    Name = "${local.stack_name} Public Subnet (AZ1)"
  }
}

resource "aws_subnet" "private_subnet1" {
  vpc_id                  = aws_vpc.vpc.id
  availability_zone       = element(data.aws_availability_zones.available.names, 0)
  cidr_block              = local.mappings["VpcConfigurations"]["cidr"]["PrivateSubnet1"]
  map_public_ip_on_launch = false
  tags = {
    "for-use-with-amazon-emr-managed-policies" = "true"
    Name                                        = "${local.stack_name} Private Subnet (AZ1)"
  }
}

resource "aws_eip" "nat_gateway1_eip" {
  vpc = true
}

resource "aws_nat_gateway" "nat_gateway1" {
  allocation_id = aws_eip.nat_gateway1_eip.id
  subnet_id     = aws_subnet.public_subnet1.id
}

resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "${local.stack_name} Public Routes"
  }
}

resource "aws_route" "default_public_route" {
  route_table_id         = aws_route_table.public_route_table.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.internet_gateway.id
}

resource "aws_route_table_association" "public_subnet1_route_table_association" {
  route_table_id = aws_route_table.public_route_table.id
  subnet_id      = aws_subnet.public_subnet1.id
}

resource "aws_route_table" "private_route_table1" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "${local.stack_name} Private Routes (AZ1)"
  }
}

resource "aws_route_table_association" "private_subnet1_route_table_association" {
  route_table_id = aws_route_table.private_route_table1.id
  subnet_id      = aws_subnet.private_subnet1.id
}

resource "aws_route" "private_subnet1_internet_route" {
  route_table_id         = aws_route_table.private_route_table1.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.nat_gateway1.id
}

resource "aws_vpc_endpoint" "s3_endpoint" {
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"
  vpc_id            = aws_vpc.vpc.id
  route_table_ids   = [aws_route_table.private_route_table1.id]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_security_group" "sage_maker_instance_security_group" {
  name        = "SMSG"
  description = "Security group with no ingress rule"
  vpc_id      = aws_vpc.vpc.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    "for-use-with-amazon-amazon-sagemaker" = "true"
  }
}

resource "aws_security_group_rule" "sage_maker_instance_security_group_ingress" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  self                     = true
  security_group_id        = aws_security_group.sage_maker_instance_security_group.id
}

resource "aws_security_group" "vpc_endpoint_security_group" {
  description = "Allow TLS for VPC Endpoint"
  vpc_id      = aws_vpc.vpc.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "${local.stack_name}-endpoint-security-group"
  }
}

resource "aws_security_group_rule" "endpoint_security_group_ingress" {
  type                     = "ingress"
  from_port                = 0
  to_port                  = 0
  protocol                 = "-1"
  source_security_group_id = aws_security_group.vpc_endpoint_security_group.id
  security_group_id        = aws_security_group.sage_maker_instance_security_group.id
}

resource "aws_iam_role" "sage_maker_execution_role" {
  name               = "${local.stack_name}-SageMakerAdminExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = ["sagemaker.amazonaws.com"]
        }
        Action = ["sts:AssumeRole"]
      }
    ]
  })
  path                 = "/"
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerPipelinesIntegrations",
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSageMakerCanvasFullAccess",
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSageMakerCanvasDataPrepFullAccess",
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonSageMakerCanvasDirectDeployAccess",
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSageMakerCanvasAIServicesAccess",
    "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonSageMakerCanvasForecastAccess"
  ]
}

resource "aws_vpc_endpoint" "vpc_endpoint_sagemaker_api" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.sagemaker.api"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_sage_maker_runtime" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.sagemaker.runtime"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_sts" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.sts"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_cw" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.monitoring"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_cwl" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.logs"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_ecr" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.ecr.dkr"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_vpc_endpoint" "vpc_endpoint_ecrapi" {
  vpc_id               = aws_vpc.vpc.id
  service_name         = "com.amazonaws.${data.aws_region.current.name}.ecr.api"
  vpc_endpoint_type    = "Interface"
  private_dns_enabled  = true
  subnet_ids           = [aws_subnet.private_subnet1.id]
  security_group_ids   = [aws_security_group.vpc_endpoint_security_group.id]
  policy               = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "*"
        Resource  = "*"
      }
    ]
  })
}

resource "aws_sagemaker_domain" "studio_domain" {
  domain_name            = var.sage_maker_domain_name
  app_network_access_type = "VpcOnly"
  auth_mode              = "IAM"
  vpc_id                 = aws_vpc.vpc.id
  subnet_ids             = [aws_subnet.private_subnet1.id]
  default_user_settings {
    execution_role  = aws_iam_role.sage_maker_execution_role.arn
    security_groups = [aws_security_group.sage_maker_instance_security_group.id]
  }
}

resource "aws_sagemaker_user_profile" "studio_user_profile" {
  domain_id         = aws_sagemaker_domain.studio_domain.id
  user_profile_name = "genai-user"
  user_settings {
    execution_role = aws_iam_role.sage_maker_execution_role.arn
  }
}

output "sage_maker_cloudformation_vpc_id" {
  description = "The ID of the Sagemaker Studio VPC"
  value       = aws_vpc.vpc.id
}

output "sage_maker_emr_demo_cloudformation_subnet_id" {
  description = "The Subnet Id of Sagemaker Studio"
  value = aws_subnet.private_subnet1.id
}

output "sage_maker_emr_demo_cloudformation_security_group" {
  description = "The Security group of Sagemaker Studio instance"
  value = aws_security_group.sage_maker_instance_security_group.arn
}

output "sage_maker_domain" {
  description = "SageMaker Domain Id"
  value = aws_sagemaker_domain.studio_domain.arn
}

output "sage_maker_domain_url" {
  description = "URL to access the SageMaker Domain"
  value = join("", ["https://console.aws.amazon.com/sagemaker/home?region=", data.aws_region.current.name, "#/studio/", aws_sagemaker_domain.studio_domain.arn])
}