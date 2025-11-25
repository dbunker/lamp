d4(o3).
d4(o1).
d3(o1).
d2(o0).
d1(o1).

d4(X) :- d4(X).
d3(X) :- d4(X), not d0(X).
d2(X) :- d3(X), not d4(X).
d1(X) :- not d3(X), d2(X).
d0(X) :- not d2(X), d1(X).