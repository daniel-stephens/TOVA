# Data sources
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-cluster"
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}-${var.environment}-api"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "web" {
  name              = "/ecs/${var.project_name}-${var.environment}-web"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "solr_api" {
  name              = "/ecs/${var.project_name}-${var.environment}-solr-api"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "solr" {
  name              = "/ecs/${var.project_name}-${var.environment}-solr"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "zookeeper" {
  name              = "/ecs/${var.project_name}-${var.environment}-zookeeper"
  retention_in_days = 7
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-${var.environment}-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Policy for EFS access (always create, EFS is always enabled)
resource "aws_iam_role_policy" "ecs_task_execution_efs" {
  name  = "${var.project_name}-${var.environment}-ecs-efs"
  role  = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "elasticfilesystem:ClientMount",
        "elasticfilesystem:ClientWrite",
        "elasticfilesystem:ClientRootAccess"
      ]
      Resource = "*"
    }]
  })
}

# IAM Role for ECS Tasks
resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-${var.environment}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

# Security Groups
resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-${var.environment}-alb-"
  vpc_id      = var.vpc_id
  description = "Security group for ALB"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-alb-sg"
  }
}

resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${var.project_name}-${var.environment}-ecs-"
  vpc_id      = var.vpc_id
  description = "Security group for ECS tasks"

  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Allow traffic from ALB"
  }

  # Allow API to connect to solr-api (allow from same security group)
  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    self        = true
    description = "Allow API to connect to solr-api"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-ecs-sg"
  }
}

# Security Group for Solr services
resource "aws_security_group" "solr_services" {
  name_prefix = "${var.project_name}-${var.environment}-solr-"
  vpc_id      = var.vpc_id
  description = "Security group for Solr and Zookeeper services"

  # Zookeeper client port
  ingress {
    from_port       = 2181
    to_port         = 2181
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "Zookeeper client port"
  }

  # Solr port
  ingress {
    from_port       = 8983
    to_port         = 8983
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "Solr port"
  }

  # solr-api port
  ingress {
    from_port       = 8001
    to_port         = 8001
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "Solr API port"
  }

  # Allow inter-service communication (self-reference using self)
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
    description = "Allow inter-service communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-solr-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = var.environment == "prod"

  tags = {
    Name = "${var.project_name}-${var.environment}-alb"
  }
}

# Target Groups
resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-${var.environment}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = {
    Name = "${var.project_name}-${var.environment}-api-tg"
  }
}

resource "aws_lb_target_group" "web" {
  name        = "${var.project_name}-${var.environment}-web-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = {
    Name = "${var.project_name}-${var.environment}-web-tg"
  }
}

# ALB Listeners
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  dynamic "default_action" {
    for_each = var.alb_certificate_arn != "" ? [1] : []
    content {
      type = "redirect"
      redirect {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }

  dynamic "default_action" {
    for_each = var.alb_certificate_arn == "" ? [1] : []
    content {
      type             = "forward"
      target_group_arn = aws_lb_target_group.web.arn
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.alb_certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.alb_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.web.arn
  }
}

# ALB Listener Rules for HTTPS
resource "aws_lb_listener_rule" "api_https" {
  count        = var.alb_certificate_arn != "" ? 1 : 0
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# ALB Listener Rules for HTTP (when no HTTPS)
resource "aws_lb_listener_rule" "api_http" {
  count        = var.alb_certificate_arn == "" ? 1 : 0
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# ECS Task Definitions
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-${var.environment}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "api"
    image = "${var.api_repo_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = concat([
      {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      },
      {
        name  = "DATABASE_URL"
        value = var.database_url
      },
      {
        name  = "DRAFTS_SAVE"
        value = "/data/drafts"
      },
      {
        name  = "API_SOLR_URL"
        value = "http://${var.project_name}-${var.environment}-solr-api.${var.project_name}.local:8001"
      }
    ], var.openai_api_key != "" ? [{
      name  = "OPENAI_API_KEY"
      value = var.openai_api_key
    }] : [])

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.api.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).getcode()==200 else 1)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_task_definition" "web" {
  family                   = "${var.project_name}-${var.environment}-web"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "web"
    image = "${var.web_repo_url}:latest"

    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "FLASK_DEBUG"
        value = var.environment == "prod" ? "0" : "1"
      },
      {
        name  = "API_BASE_URL"
        value = var.alb_certificate_arn != "" ? "https://${var.domain_name != "" ? var.domain_name : aws_lb.main.dns_name}/api" : "http://${aws_lb.main.dns_name}/api"
      },
      {
        name  = "DATABASE_URL"
        value = var.database_url
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.web.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# ECS Services
resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-${var.environment}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [
    aws_lb_listener.http
  ]

  tags = {
    Name = "${var.project_name}-${var.environment}-api"
  }
}

resource "aws_ecs_service" "web" {
  name            = "${var.project_name}-${var.environment}-web"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.web.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.web.arn
    container_name   = "web"
    container_port   = 8080
  }

  depends_on = [
    aws_lb_listener.http
  ]

  tags = {
    Name = "${var.project_name}-${var.environment}-web"
  }
}

# Service Discovery for internal service communication
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${var.project_name}.local"
  description = "Service discovery namespace for ${var.project_name}"
  vpc         = var.vpc_id
}

# EFS Access Points for persistent storage
# Note: EFS is always created, so we can always create access points
resource "aws_efs_access_point" "solr" {
  file_system_id = var.efs_file_system_id

  posix_user {
    gid = 8983
    uid = 8983
  }

  root_directory {
    path = "/solr"
    creation_info {
      owner_gid   = 8983
      owner_uid   = 8983
      permissions = "755"
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-solr-ap"
  }
}

resource "aws_efs_access_point" "zookeeper" {
  file_system_id = var.efs_file_system_id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/zookeeper"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-zookeeper-ap"
  }
}

# Local values for EFS access point IDs
locals {
  efs_access_point_solr       = aws_efs_access_point.solr.id
  efs_access_point_zookeeper   = aws_efs_access_point.zookeeper.id
}

# Zookeeper Task Definition
resource "aws_ecs_task_definition" "zookeeper" {
  family                   = "${var.project_name}-${var.environment}-zookeeper"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "zookeeper"
    image = "zookeeper:3.9"

    portMappings = [
      {
        containerPort = 2181
        protocol      = "tcp"
      },
      {
        containerPort = 8080
        protocol      = "tcp"
      }
    ]

    environment = [
      {
        name  = "JVMFLAGS"
        value = "-Djute.maxbuffer=50000000"
      }
    ]

    mountPoints = [{
      sourceVolume  = "zookeeper-data"
      containerPath = "/data"
    }, {
      sourceVolume  = "zookeeper-data"
      containerPath = "/datalog"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.zookeeper.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  volume {
    name = "zookeeper-data"
    efs_volume_configuration {
      file_system_id     = var.efs_file_system_id
      root_directory     = "/"
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = local.efs_access_point_zookeeper
        iam             = "ENABLED"
      }
    }
  }
}

# Solr Task Definition
resource "aws_ecs_task_definition" "solr" {
  family                   = "${var.project_name}-${var.environment}-solr"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "solr"
    image = "${var.solr_repo_url}:latest"

    portMappings = [{
      containerPort = 8983
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "ZK_HOST"
        value = "${var.project_name}-${var.environment}-zookeeper:2181"
      }
    ]

    command = [
      "solr",
      "-f",
      "-c",
      "-z",
      "${var.project_name}-${var.environment}-zookeeper:2181",
      "-a",
      "-Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=1044 -Djute.maxbuffer=0x5000000"
    ]

    mountPoints = [{
      sourceVolume  = "solr-data"
      containerPath = "/var/solr"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.solr.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  volume {
    name = "solr-data"
    efs_volume_configuration {
      file_system_id     = var.efs_file_system_id
      root_directory     = "/"
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = local.efs_access_point_solr
        iam             = "ENABLED"
      }
    }
  }
}

# Solr API Task Definition
resource "aws_ecs_task_definition" "solr_api" {
  family                   = "${var.project_name}-${var.environment}-solr-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "solr-api"
    image = "${var.solr_api_repo_url}:latest"

    portMappings = [{
      containerPort = 8001
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "SOLR_URL"
        value = "http://${var.project_name}-${var.environment}-solr:8983"
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.solr_api.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# Zookeeper Service
resource "aws_ecs_service" "zookeeper" {
  name            = "${var.project_name}-${var.environment}-zookeeper"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.zookeeper.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.solr_services.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.zookeeper.arn
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-zookeeper"
  }
}

# Solr Service
resource "aws_ecs_service" "solr" {
  name            = "${var.project_name}-${var.environment}-solr"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.solr.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.solr_services.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.solr.arn
  }

  depends_on = [aws_ecs_service.zookeeper]

  tags = {
    Name = "${var.project_name}-${var.environment}-solr"
  }
}

# Solr API Service
resource "aws_ecs_service" "solr_api" {
  name            = "${var.project_name}-${var.environment}-solr-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.solr_api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.solr_services.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.solr_api.arn
  }

  depends_on = [aws_ecs_service.solr]

  tags = {
    Name = "${var.project_name}-${var.environment}-solr-api"
  }
}

# Service Discovery Services
resource "aws_service_discovery_service" "zookeeper" {
  name = "${var.project_name}-${var.environment}-zookeeper"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

}

resource "aws_service_discovery_service" "solr" {
  name = "${var.project_name}-${var.environment}-solr"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

}

resource "aws_service_discovery_service" "solr_api" {
  name = "${var.project_name}-${var.environment}-solr-api"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

}

