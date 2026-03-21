%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.415s
%% Conflict analysis                       : 2.307s
%%   - Positive Examples                   : 2.307s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.874s
%% Total                                   : 3.703s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d6(V1) :- d4(V1).
d8(V1) :- d5(V1).
d9(V1) :- d8(V1).
d7(V1) :- d8(V1).
d1(V1) :- d9(V1); not d8(V1).

