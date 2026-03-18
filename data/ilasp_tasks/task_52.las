#constant(obj, o2).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d2(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d0(o2), d1(o4), d2(o2)
}, {
  d0(o4), d0(o5), d0(o9), d1(o2), d1(o5), d1(o9), d2(o4), d2(o5), d2(o9)
}, {
  d3(o2).
  d3(o5).
  d3(o9).
  d4(o4).
  d5(o2).
  d5(o4).
}).