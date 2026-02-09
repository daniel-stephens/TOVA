# EFS File System for persistent storage
resource "aws_efs_file_system" "main" {
  creation_token = "${var.project_name}-${var.environment}-efs"
  
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"
  encrypted        = true

  tags = {
    Name = "${var.project_name}-${var.environment}-efs"
  }
}

# EFS Mount Targets (one per availability zone)
resource "aws_efs_mount_target" "main" {
  count           = length(var.private_subnet_ids)
  file_system_id  = aws_efs_file_system.main.id
  subnet_id       = var.private_subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

# Security Group for EFS
resource "aws_security_group" "efs" {
  name_prefix = "${var.project_name}-${var.environment}-efs-"
  vpc_id      = var.vpc_id
  description = "Security group for EFS"

  dynamic "ingress" {
    for_each = length(var.allowed_security_group_ids) > 0 ? var.allowed_security_group_ids : []
    content {
      from_port       = 2049
      to_port         = 2049
      protocol        = "tcp"
      security_groups = [ingress.value]
      description     = "NFS access from ECS tasks"
    }
  }

  # Allow NFS from VPC CIDR if no specific security groups provided
  dynamic "ingress" {
    for_each = length(var.allowed_security_group_ids) == 0 ? [1] : []
    content {
      from_port   = 2049
      to_port     = 2049
      protocol    = "tcp"
      cidr_blocks = [var.vpc_cidr]
      description = "NFS access from VPC"
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-efs-sg"
  }
}

