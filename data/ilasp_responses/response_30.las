%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.654s
%% Conflict analysis                       : 2.842s
%%   - Positive Examples                   : 2.842s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.67s
%% Total                                   : 4.325s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d6(V1) :- d5(V1).
d2(V1) :- d5(V1).
d8(V1) :- d7(V1); not d0(V1).
d7(V1) :- d0(V1); not d5(V1).

