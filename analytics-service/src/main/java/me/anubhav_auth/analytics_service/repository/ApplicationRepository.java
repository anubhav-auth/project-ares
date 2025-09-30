package me.anubhav_auth.analytics_service.repository;

import me.anubhav_auth.analytics_service.entity.Application;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Repository
public interface ApplicationRepository extends JpaRepository<Application, Long> {

    // Count methods
    long countByStatus(String status);
    long countByCreatedAtAfter(Instant date);

    // Find methods
    List<Application> findByStatus(String status);
    List<Application> findByCompanyNameContainingIgnoreCase(String companyName);
    Optional<Application> findByJobId(String jobId);

    // Check for duplicates
    boolean existsByCompanyNameAndJobTitleAndCreatedAtAfter(
            String companyName,
            String jobTitle,
            Instant date
    );

    // Custom queries
    @Query("SELECT a FROM Application a WHERE a.status = :status ORDER BY a.createdAt DESC")
    List<Application> findRecentByStatus(@Param("status") String status);

    @Query("SELECT a FROM Application a WHERE a.createdAt >= :startDate AND a.createdAt <= :endDate")
    List<Application> findApplicationsBetweenDates(
            @Param("startDate") Instant startDate,
            @Param("endDate") Instant endDate
    );

    @Query("SELECT COUNT(a) FROM Application a WHERE DATE(a.createdAt) = CURRENT_DATE")
    long countTodayApplications();

    // For cleanup
    void deleteByCreatedAtBefore(Instant date);
}