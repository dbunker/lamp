%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.2s
%% Conflict analysis                       : 0.67s
%%   - Positive Examples                   : 0.67s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.157s
%% Total                                   : 1.065s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d1(V1).
d5(V1) :- d1(V1).
d2(V1) :- d5(V1).
d4(V1) :- d3(V1).

