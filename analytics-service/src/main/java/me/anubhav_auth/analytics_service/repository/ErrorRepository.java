package me.anubhav_auth.analytics_service.repository;

import me.anubhav_auth.analytics_service.entity.WorkflowError;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;

@Repository
public interface ErrorRepository extends JpaRepository<WorkflowError, Long> {

    List<WorkflowError> findByOccurredAtAfterOrderByOccurredAtDesc(Instant date);

    List<WorkflowError> findByJobId(String jobId);

    List<WorkflowError> findByNodeName(String nodeName);

    @Modifying
    @Transactional
    void deleteByOccurredAtBefore(Instant date);

    long countByOccurredAtAfter(Instant date);
}