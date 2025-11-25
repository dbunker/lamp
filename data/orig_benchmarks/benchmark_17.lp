d0(o1).
d0(o2).
d3(o0).
d3(o2).

d1(X) :- not d2(X), not d0(X), d3(X).
d2(X) :- not d0(X), d1(X), not d3(X).
d1(X) :- not d0(X), d2(X), not d3(X).