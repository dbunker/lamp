%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.076s
%% Conflict analysis                       : 23.983s
%%   - Positive Examples                   : 23.983s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 9.191s
%% Total                                   : 36.275s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d4(V1).
d9(V1) :- d4(V1).
d2(V1) :- d4(V1).
d4(V1) :- d0(V1).
d7(V1) :- d4(V1).
d3(V1) :- d4(V1).

