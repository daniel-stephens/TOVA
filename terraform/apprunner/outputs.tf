output "api_url" {
  value = aws_apprunner_service.api.service_url
}

output "web_url" {
  value = aws_apprunner_service.web.service_url
}


output "rds_endpoint" {
  value     = module.rds.db_endpoint
  sensitive = true
}

output "ecr_repository_urls" {
  value = {
    api     = aws_ecr_repository.api.repository_url
    web     = aws_ecr_repository.web.repository_url
    solr_api = aws_ecr_repository.solr_api.repository_url
  }
}

