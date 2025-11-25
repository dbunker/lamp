d1(o4).
d2(o0).
d3(o2).
d0(o0).

d4(X) :- not d1(X), d2(X), not d3(X).
d0(X) :- not d2(X), d1(X), not d3(X).