describe('Result page: header new transcription button', () => {
  it('is visible when result is shown', () => {
    cy.visit('/');

    // Simulate showing a result by manipulating the DOM
    cy.get('#result').invoke('removeClass', 'hidden');
    cy.get('#result-text').invoke('text', 'Hello world test transcription');

    // Header new transcription button should still be visible
    cy.get('#header-new-btn').should('be.visible');
    cy.get('#header-new-btn').should('not.be.disabled');

    // Click should navigate to /
    cy.get('#header-new-btn').click();
    cy.location('pathname').should('eq', '/');
  });
});
