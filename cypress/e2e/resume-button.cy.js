describe('Resume page: header and home buttons', () => {
  it('header-new-btn is visible and resume-home-btn navigates to /', () => {
    // Visit a nonexistent job page (valid 32-char hex UUID format)
    cy.visit('/j/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');

    // The resume view should be visible (job not found)
    cy.get('#resume-view').should('be.visible');
    cy.get('#resume-error').should('be.visible');

    // Header new transcription button is still visible
    cy.get('#header-new-btn').should('exist');
    cy.get('#header-new-btn').should('be.visible');
    cy.get('#header-new-btn').should('not.be.disabled');

    // Resume home button appears and is clickable
    cy.get('#resume-home-btn').should('be.visible');
    cy.get('#resume-home-btn').should('not.be.disabled');

    // Click header button navigates to /
    cy.get('#header-new-btn').click();
    cy.location('pathname').should('eq', '/');
  });
});
