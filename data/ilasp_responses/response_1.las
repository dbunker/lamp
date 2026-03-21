%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 2.019s
%% Conflict analysis                       : 197.471s
%%   - Positive Examples                   : 197.471s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 27.625s
%% Total                                   : 228.444s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d2(V1).
d1(V1) :- d3(V1).
d9(V1) :- d5(V1).
d0(V1) :- d1(V1); not d2(V1).
d1(V1) :- d9(V1); not d4(V1); not d7(V1).
d6(V1) :- d9(V1); not d5(V1); not d7(V1).

