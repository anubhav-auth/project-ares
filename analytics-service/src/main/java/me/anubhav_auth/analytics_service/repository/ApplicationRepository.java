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

    Optional<Application> findByJobId(String jobId);
    boolean existsByJobId(String jobId);

    boolean existsByCompanyNameAndJobTitleAndCreatedAtAfter(
            String companyName,
            String jobTitle,
            Instant date
    );

    List<Application> findByCompanyNameAndJobTitleAndCreatedAtAfter(
            String companyName,
            String jobTitle,
            Instant date
    );

    @Query("SELECT a FROM Application a WHERE " +
            "(a.jobId = :jobId) OR " +
            "(a.companyName = :companyName AND a.jobTitle = :jobTitle)")
    List<Application> findPotentialDuplicates(
            @Param("jobId") String jobId,
            @Param("companyName") String companyName,
            @Param("jobTitle") String jobTitle
    );

    List<Application> findByStatus(String status);
    long countByStatus(String status);

    long countByCreatedAtAfter(Instant date);

    List<Application> findByCompanyNameContainingIgnoreCase(String companyName);

    void deleteByCreatedAtBeforeAndStatus(Instant date, String status);
}