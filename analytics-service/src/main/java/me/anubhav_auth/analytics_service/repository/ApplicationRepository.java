package me.anubhav_auth.analytics_service.repository;

import me.anubhav_auth.analytics_service.entity.Application;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository // Marks this as a Spring Data repository
public interface ApplicationRepository extends JpaRepository<Application, Long> {
    // By extending JpaRepository, we get methods like save(), findById(), findAll() for free!
}