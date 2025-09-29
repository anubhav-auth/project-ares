package me.anubhav_auth.analytics_service.controller;

import me.anubhav_auth.analytics_service.entity.Application;
import me.anubhav_auth.analytics_service.repository.ApplicationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/applications")
public class ApplicationController {

    @Autowired
    private ApplicationRepository applicationRepository;

    @PostMapping
    public Application createApplication(@RequestBody Application application) {
        return applicationRepository.save(application);
    }

    @GetMapping
    public List<Application> getAllApplications() {
        return applicationRepository.findAll();
    }

    @PutMapping("/{id}")
    public ResponseEntity<Application> updateApplicationStatus(
            @PathVariable Long id,
            @RequestBody Application applicationDetails) {
        return applicationRepository.findById(id)
                .map(application -> {
                    application.setStatus(applicationDetails.getStatus()); // Only update the status
                    Application updatedApplication = applicationRepository.save(application);
                    return ResponseEntity.ok(updatedApplication);
                })
                .orElse(ResponseEntity.notFound().build());
    }
}