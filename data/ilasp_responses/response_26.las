%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 1                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Iteration 2                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 1.396s
%% Conflict analysis                       : 39.211s
%%   - Positive Examples                   : 39.211s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 15.905s
%% Total                                   : 57.495s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Searching for counterexample... 
%% Found positive counterexample: eg1 (a total of 1 counterexamples found) 
%% Found hypothesis: [32, 89, 94, 107, 145, 160, 2309] 15 
%% d1(V1) :- d8(V1). 
%% d2(V1) :- d7(V1). 
%% d3(V1) :- d7(V1). 
%% d8(V1) :- d7(V1). 
%% d9(V1) :- d5(V1). 
%% d3(V1) :- d4(V1). 
%% d0(V1) :- d2(V1); not d7(V1). 
%% Searching for counterexample... 
%%  
%%  
%% Final Hypothesis: 
%%  
d1(V1) :- d8(V1).
d2(V1) :- d7(V1).
d3(V1) :- d7(V1).
d8(V1) :- d7(V1).
d9(V1) :- d5(V1).
d3(V1) :- d4(V1).
d0(V1) :- d2(V1); not d7(V1).

