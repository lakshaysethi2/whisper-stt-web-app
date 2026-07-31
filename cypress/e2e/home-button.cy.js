describe('Home page: header new transcription button', () => {
  it('is present, visible, clickable, and navigates to /', () => {
    cy.visit('/');

    // Button exists and is visible
    cy.get('#header-new-btn').should('exist');
    cy.get('#header-new-btn').should('be.visible');

    // Button is clickable
    cy.get('#header-new-btn').should('not.be.disabled');

    // Click triggers navigation to /
    cy.get('#header-new-btn').click();
    cy.location('pathname').should('eq', '/');
  });
});
