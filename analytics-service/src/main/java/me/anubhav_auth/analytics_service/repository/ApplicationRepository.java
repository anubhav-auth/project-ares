package me.anubhav_auth.analytics_service.repository;

import me.anubhav_auth.analytics_service.entity.Application;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ApplicationRepository extends JpaRepository<Application, Long> {
}