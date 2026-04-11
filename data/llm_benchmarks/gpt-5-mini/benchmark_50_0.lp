d1(o0).
d2(o7).
d2(o0).
d1(o1).
d4(o4).
d3(o0).

d0(X) :- d1(X), not d2(X).
d3(X) :- d4(X).
d2(X) :- d4(X).
d1(X) :- d4(X).